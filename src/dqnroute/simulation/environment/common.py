from simpy import Environment, Event, Interrupt
from ...agents import *
from ...utils import *

from ..factory.common import HandlerFactory


class UnknownAgentError(Exception):
    pass


class MultiAgentEnv(HasLog):
    """
    Abstract class which simulates an environment with multiple agents,
    where agents are connected accordingly to a given connection graph.
    """
    def __init__(self, env: Environment, **kwargs):
        self.env = env
        self.conn_graph = self.makeConnGraph(**kwargs)
        self.factory = self.makeHandlerFactory(
            env=self.env,
            conn_graph=self.conn_graph,
            **kwargs
        )

        agent_ids = list(self.conn_graph.nodes)
        self.handlers = {agent_id: self.factory._makeHandler(agent_id) for agent_id in agent_ids}
        self.delayed_evs = {agent_id: {} for agent_id in agent_ids}

        self.master_agent = ('master', 0)
        if self.factory.centralized():
            self.handlers[('master', 0)] = self.factory.master_handler
            self.delayed_evs[('master', 0)] = {}

        self._agent_passes = 0

    def time(self):
        return self.env.now

    def logName(self):
        return 'World'

    def makeConnGraph(self, **kwargs) -> nx.Graph:
        """
        A method which defines a connection graph for the system with
        given params.
        Should be overridden. The node labels of a resulting graph should be
        `AgentId`s.
        """
        raise NotImplementedError()

    def makeHandlerFactory(self, **kwargs) -> HandlerFactory:
        """
        Makes a handler factory
        """
        raise NotImplementedError()

    def handle(self, from_agent: AgentId, event: WorldEvent) -> Event:
        # print(f'handle, from: {from_agent}, event: {event}')
        """
        Main method which governs how events cause each other in the
        environment. Not to be overridden in children: `handleAction` and
        `handleWorldEvent` should be overridden instead.
        """
        if isinstance(event, MasterEvent):
            from_agent = event.agent
            event = event.inner

        if isinstance(event, Message):
            return self.handleMessage(from_agent, event)

        elif isinstance(event, Action):
            return self.handleAction(from_agent, event)

        elif isinstance(event, DelayedEvent):
            proc = self.env.process(self._delayedHandleGen(from_agent, event))
            self.delayed_evs[from_agent][event.id] = proc
            return Event(self.env).succeed()

        elif isinstance(event, DelayInterrupt):
            try:
                self.delayed_evs[from_agent][event.delay_id].interrupt()
            except (KeyError, RuntimeError):
                pass
            return Event(self.env).succeed()

        elif from_agent[0] == 'world':
            return self.handleWorldEvent(event)

        else:
            raise Exception('Non-world event: ' + str(event))

    def handleMessage(self, from_agent: AgentId, msg: Message) -> Event:
        """
        Method which handles how messages should be dealt with. Is not meant to be
        overridden.
        """
        if isinstance(msg, WireOutMsg):
            # Out message is considered to be handled as soon as its
            # handling by the recipient is scheduled. We do not
            # wait for other agent to handle them.
            self.env.process(self._handleOutMsgGen(from_agent, msg))
            return Event(self.env).succeed()
        else:
            raise UnsupportedMessageType(msg)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        """
        Method which governs how agents' actions influence the environment
        Should be overridden by subclasses.
        """
        raise UnsupportedActionType(action)

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        """
        Method which governs how events from outside influence the environment.
        Should be overridden by subclasses.
        """
        if isinstance(event, LinkUpdateEvent):
            return self.handleConnGraphChange(event)
        else:
            raise UnsupportedEventType(event)

    def passToAgent(self, agent: AgentId, event: WorldEvent) -> Event:
        """
        Let an agent react on event and handle all events produced by agent as
        a consequence.
        """
        # print('Env: pass to agent', agent, event, self.handlers[agent], self.handlers[agent].id)
        self._agent_passes += 1
        if self.factory.centralized() and isinstance(self.factory.master_handler, Oracle):
            agent_evs = delayed_first(self.factory.master_handler.handle(event))
        elif agent in self.handlers:
            agent_evs = delayed_first(self.handlers[agent].handle(event))
        else:
            raise UnknownAgentError(f'No such agent: {agent}')

        evs = []
        for new_event in agent_evs:
            # print('handling event in env, agent:', agent, 'new_event:', new_event)
            evs.append(self.handle(agent, new_event))
        return self.env.all_of(evs)

    def handleConnGraphChange(self, event: LinkUpdateEvent) -> Event:
        """
        Adds or removes the connection link and notifies the agents that
        the corresponding interfaces changed availability.
        Connection graph itself does not change to preserve interfaces numbering.
        """
        u = event.u
        v = event.v
        u_int = interface_idx(self.conn_graph, u, v)
        v_int = interface_idx(self.conn_graph, v, u)

        if isinstance(event, AddLinkEvent):
            u_ev = InterfaceSetupEvent(u_int, v, event.params)
            v_ev = InterfaceSetupEvent(v_int, u, event.params)
        elif isinstance(event, RemoveLinkEvent):
            u_ev = InterfaceShutdownEvent(u_int)
            v_ev = InterfaceShutdownEvent(v_int)
        return self.passToAgent(u, u_ev) & self.passToAgent(v, v_ev)

    def _delayedHandleGen(self, from_agent: AgentId, event: DelayedEvent):
        proc_id = event.id
        delay = event.delay
        inner = event.inner

        try:
            yield self.env.timeout(delay)
            self.handle(from_agent, inner)
        except Interrupt:
            pass
        del self.delayed_evs[from_agent][proc_id]

    def _handleOutMsgGen(self, from_agent: AgentId, msg: WireOutMsg):
        int_id = msg.interface
        inner = msg.payload
        to_agent, to_interface = resolve_interface(self.conn_graph, from_agent, int_id)
        yield self.passToAgent(to_agent, WireInMsg(to_interface, inner))
