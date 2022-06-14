from simpy import Environment, Event, Interrupt
from ...agents import *
from ...utils import *


class HandlerFactory:
    def __init__(self, env: Environment, conn_graph: nx.Graph, topology_graph=None, **kwargs):
        super().__init__()
        self.env = env
        self.conn_graph = conn_graph

        self.topology_graph = topology_graph
        if topology_graph is None:
            self.topology_graph = self.conn_graph.to_directed()

        if self.centralized():
            self.master_handler = self.makeMasterHandler()

    def _makeHandler(self, agent_id: AgentId, **kwargs) -> MessageHandler:
        neighbours = [v for _, v in self.conn_graph.edges(agent_id)]
        if self.centralized():
            return SlaveHandler(id=agent_id, master=self.master_handler, neighbors=neighbours)
        else:
            return self.makeHandler(agent_id, neighbours=neighbours, **kwargs)

    def makeMasterHandler(self) -> MasterHandler:
        raise NotImplementedError()

    def makeHandler(self, agent_id: AgentId) -> MessageHandler:
        raise NotImplementedError()

    def handlerClass(self, handler_type: str):
        raise NotImplementedError()

    def centralized(self):
        raise NotImplementedError()
