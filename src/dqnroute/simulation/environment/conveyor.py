from ...constants import *
from ..common import *
from ..factory.router import RouterFactory
from ..factory.conveyor import ConveyorFactory
from .common import MultiAgentEnv


DIVERTER_RANGE = 0.5
REAL_BAG_RADIUS = 0.5


class ConveyorsEnvironment(MultiAgentEnv):
    """
    Environment which models the conveyor system and the movement of bags.
    """

    def __init__(self, env: Environment, conveyors_layout, data_series: EventSeries,
                 speed: float = 1, energy_consumption: float = 1,
                 default_conveyor_args={}, default_router_args={}, **kwargs):
        self.max_speed = speed
        self.energy_consumption = energy_consumption
        self.layout = conveyors_layout
        self.data_series = data_series

        self.topology_graph = make_conveyor_topology_graph(conveyors_layout)

        # Initialize conveyor-wise state dictionaries
        conv_ids = list(self.layout['conveyors'].keys())
        dyn_env = DynamicEnv(time=lambda: self.env.now)

        self.conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes(self.topology_graph, conv_id,
                                             only_own=True, data='conveyor_pos')
            length = self.layout['conveyors'][conv_id]['length']
            model = ConveyorModel(dyn_env, length, self.max_speed,
                                  checkpoints, self.data_series.subSeries('energy'),
                                  model_id=('world_conv', conv_id))
            self.conveyor_models[conv_id] = model

        self.conveyors_move_proc = None
        self.current_bags = {}
        self.conveyor_broken = {conv_id: False for conv_id in conv_ids}

        self.conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self.topology_graph, conv_id)[-1]
            for conv_id in conv_ids
        }

        super().__init__(
            env=env, topology=self.topology_graph, conveyors_layout=conveyors_layout,
            energy_consumption=energy_consumption, max_speed=speed, **kwargs)

        self._updateAll()

    def log(self, msg, force=False):
        if force:
            super().log(msg, True)

    def makeHandlerFactory(self, **kwargs):
        kwargs['conveyor_models'] = self.conveyor_models
        return ConveyorFactory(**kwargs)

    def makeConnGraph(self, conveyors_layout, **kwargs) -> nx.Graph:
        return make_conveyor_conn_graph(conveyors_layout)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        if isinstance(action, BagReceiveAction):
            assert agent_type(from_agent) == 'sink', "Only sink can receive bags!"
            bag = action.bag

            self.log(f"bag #{bag.id} received at sink {from_agent[1]}")

            if from_agent != bag.dst:
                raise Exception(f'Bag #{action.bag.id} came to {from_agent}, but its destination was {bag.dst}')

            assert bag.id in self.current_bags, "why leave twice??"
            self.current_bags.pop(action.bag.id)

            # fix to make reinforce work
            from ...agents.routers.reinforce import PackageHistory
            PackageHistory.finishHistory(bag)

            self.data_series.logEvent('time', self.env.now, self.env.now - action.bag.start_time)
            return Event(self.env).succeed()

        elif isinstance(action, DiverterKickAction):
            assert agent_type(from_agent) == 'diverter', "Only diverter can do kick actions!"
            self.log(f'diverter {agent_idx(from_agent)} kicks')

            return self._checkInterrupt(lambda: self._diverterKick(from_agent))

        elif isinstance(action, ConveyorSpeedChangeAction):
            assert agent_type(from_agent) == 'conveyor', "Only conveyor can change speed!"
            self.log(f'change conv {agent_idx(from_agent)} speed to {action.new_speed}')

            return self._checkInterrupt(
                lambda: self._changeConvSpeed(agent_idx(from_agent), action.new_speed))

        else:
            return super().handleAction(from_agent, action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, BagAppearanceEvent):
            src = ('source', event.src_id)
            bag = event.bag
            self.current_bags[bag.id] = set()
            self.log(f"bag #{bag.id} appeared at source {src[1]}")

            conv_idx = conveyor_idx(self.topology_graph, src)
            self.passToAgent(src, BagDetectionEvent(bag))
            return self._checkInterrupt(lambda: self._putBagOnConveyor(conv_idx, src, bag, src))

        elif isinstance(event, ConveyorBreakEvent):
            return self._checkInterrupt(lambda: self._conveyorBreak(event.conv_idx))
        elif isinstance(event, ConveyorRestoreEvent):
            return self._checkInterrupt(lambda: self._conveyorRestore(event.conv_idx))
        else:
            return super().handleWorldEvent(event)

    def _checkInterrupt(self, callback):
        if self.conveyors_move_proc is None:
            callback()
        else:
            try:
                self.conveyors_move_proc.interrupt()
                self.conveyors_move_proc = None
            except RuntimeError as err:
                self.log(f'UNEXPECTED INTERRUPT FAIL {err}', True)

            for model in self.conveyor_models.values():
                model.pause()

            callback()
            self._updateAll()

        return Event(self.env).succeed()

    def _conveyorBreak(self, conv_idx: int):
        self.log(f'conv break: {conv_idx}', True)
        model = self.conveyor_models[conv_idx]
        model.setSpeed(0)
        self.log(f'chill bags: {len(model.objects)}', True)

        self.conveyor_broken[conv_idx] = True
        for aid in self.handlers.keys():
            self.passToAgent(aid, ConveyorBreakEvent(conv_idx))

    def _conveyorRestore(self, conv_idx: int):
        self.log(f'conv restore: {conv_idx}', True)
        self.conveyor_broken[conv_idx] = False
        for aid in self.handlers.keys():
            self.passToAgent(aid, ConveyorRestoreEvent(conv_idx))

    def _diverterKick(self, dv_id: AgentId):
        """
        Checks if some bag is in front of a given diverter now,
        if so, moves this bag from current conveyor to upstream one.
        """
        assert agent_type(dv_id) == 'diverter', "only diverter can kick!!"

        dv_idx = agent_idx(dv_id)
        dv_cfg = self.layout['diverters'][dv_idx]
        conv_idx = dv_cfg['conveyor']
        up_conv = dv_cfg['upstream_conv']
        pos = dv_cfg['pos']

        conv_model = self.conveyor_models[conv_idx]
        n_bag, n_pos = conv_model.nearestObject(pos)

        if abs(pos - n_pos) <= DIVERTER_RANGE:
            self._removeBagFromConveyor(conv_idx, n_bag.id, dv_id)
            self._putBagOnConveyor(up_conv, dv_id, n_bag, dv_id)

    def _changeConvSpeed(self, conv_idx: int, new_speed: float):
        """
        Changes the conveyor speed, updating all the bag movement processes
        accordingly. If the speed just became non-zero, then the conveyor became working;
        if it just became zero, then the conveyor stopped.
        """
        model = self.conveyor_models[conv_idx]
        old_speed = model.speed
        if new_speed == old_speed:
            return

        model.setSpeed(new_speed)

        if old_speed == 0:
            self.log(f'conv {conv_idx} started!')
        if new_speed == 0:
            self.log(f'conv {conv_idx} stopped!')

    def _removeBagFromConveyor(self, conv_idx, bag_id, node):
        model = self.conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        conv_aid = ('conveyor', conv_idx)
        self.passToAgent(conv_aid, OutgoingBagEvent(bag, node))
        return bag

    def _putBagOnConveyor(self, conv_idx, sender, bag, node):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """
        if self.conveyor_broken[conv_idx]:
            # just forget about the bag and say we had a collision
            self.data_series.logEvent('collisions', self.env.now, 1)
            self.current_bags.pop(bag.id)
            return

        pos = node_conv_pos(self.topology_graph, conv_idx, node)
        assert pos is not None, "adasdasdasdas!"

        self.log(f'bag {bag.id} -> conv {conv_idx} ({pos}m)')
        model = self.conveyor_models[conv_idx]
        nearest = model.putObject(bag.id, bag, pos, return_nearest=True)
        if nearest is not None:
            n_oid, n_pos = nearest
            if abs(pos - n_pos) < 2 * REAL_BAG_RADIUS:
                self.log(f'collision detected: (#{bag.id}; {pos}m) with (#{n_oid}; {n_pos}m) on conv {conv_idx}',
                         True)
                self.data_series.logEvent('collisions', self.env.now, 1)

        bag.last_conveyor = conv_idx
        conv_aid = ('conveyor', conv_idx)

        # added by Igor
        # to patch the situation of a bag passing more than one time through the same conveyor
        # remove all the nodes of this conveyor
        # print(node)
        # print(dir(self))
        # assert False
        self.current_bags[bag.id] = set()

        self.current_bags[bag.id].add(node)
        self.passToAgent(conv_aid, IncomingBagEvent(sender, bag, node))

    def _leaveConveyorEnd(self, conv_idx, bag_id) -> bool:
        bag = self._removeBagFromConveyor(conv_idx, bag_id, ('conv_end', conv_idx))
        up_node = self.conveyor_upstreams[conv_idx]
        up_type = agent_type(up_node)

        if up_type == 'sink':
            self.passToAgent(up_node, BagDetectionEvent(bag))
            return True

        if up_type == 'junction':
            up_conv = conveyor_idx(self.topology_graph, up_node)
            self._putBagOnConveyor(up_conv, ('conveyor', conv_idx), bag, up_node)
        else:
            raise Exception('Invalid conveyor upstream node type: ' + up_type)
        return False

    def _updateAll(self):
        self.log('CHO PO')
        self.conveyors_move_proc = None

        left_to_sinks = set()
        # Resolving all immediate events
        for (conv_idx, (bag, node, delay)) in all_unresolved_events(self.conveyor_models):
            assert delay == 0, "well that's just obnoxious"
            if self.conveyor_broken[conv_idx]:
                continue

            # print(f"BEFORE ~ event {(bag, node, delay)} on conv {conv_idx}; {bag.id in left_to_sinks}; {node in self.current_bags[bag.id]}")
            if bag.id in left_to_sinks or node in self.current_bags[bag.id]:
                continue

            # print(f"AFTER ~ event {(bag, node, delay)} on conv {conv_idx}")

            self.log(f'conv {conv_idx}: handling {bag} on {node}')

            model = self.conveyor_models[conv_idx]
            atype = agent_type(node)
            left_to_sink = False

            if atype == 'junction':
                self.passToAgent(('conveyor', conv_idx), PassedBagEvent(bag, node))
            elif atype == 'conv_end':
                left_to_sink = self._leaveConveyorEnd(conv_idx, bag.id)
                if left_to_sink:
                    left_to_sinks.add(bag.id)
            elif atype == 'diverter':
                self.passToAgent(node, BagDetectionEvent(bag))
                if bag.id in model.objects:
                    self.passToAgent(('conveyor', conv_idx), PassedBagEvent(bag, node))
            else:
                raise Exception(f'Impossible conv node: {node}')

            if bag.id in self.current_bags and bag.id not in left_to_sinks:
                self.current_bags[bag.id].add(node)

        for conv_idx, model in self.conveyor_models.items():
            if model.resolving():
                model.endResolving()
            model.resume()

        self.conveyors_move_proc = self.env.process(self._move())

    def _move(self):
        try:
            events = all_next_events(self.conveyor_models)
            self.log(f'MOVING: {events}')

            if len(events) > 0:
                conv_idx, (bag, node, delay) = events[0]
                assert delay > 0, "next event delay is 0!"
                self.log(f'NEXT EVENT: conv {conv_idx} - ({bag}, {node}, {delay})')
                yield self.env.timeout(delay)
            else:
                # hang forever (until interrupt)
                yield Event(self.env)

            for model in self.conveyor_models.values():
                model.pause()

            self._updateAll()
        except Interrupt:
            pass