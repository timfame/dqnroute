import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import *
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sympy

import os
current_dir = os.getcwd()
os.chdir("../src")
from dqnroute import *
from dqnroute.networks import *
from dqnroute.verification.router_graph import RouterGraph
from dqnroute.verification.adversarial import PGDAdversary
from dqnroute.verification.ml_util import Util
from dqnroute.verification.markov_analyzer import MarkovAnalyzer
os.chdir(current_dir)


parser = argparse.ArgumentParser(description="Verifier of baggage routing neural networks.")
parser.add_argument("--command", type=str, required=True,
                    help="one of deterministic_test, embedding_adversarial, q_adversarial, q_adversarial_lipschitz, compare")
parser.add_argument("--config_file", type=str, required=True,
                    help="YAML config file with the topology graph and other configuration info")
parser.add_argument("--probability_smoothing", type=float, default=0.01,
                    help="smoothing (0..1) of probabilities during learning and verification (defaut: 0.01)")
parser.add_argument("--random_seed", type=int, default=42,
                    help="random seed for pretraining and training (default: 42)")
parser.add_argument("--force_pretrain", action="store_true",
                    help="whether not to load previously saved pretrained models and force recomputation")
parser.add_argument("--force_train", action="store_true",
                    help="whether not to load previously saved trained models and force recomputation")
parser.add_argument("--simple_path_cost", action="store_true",
                    help="use the number of transitions instead of the total conveyor length as path cost")
parser.add_argument("--skip_graphviz", action="store_true",
                    help="do not visualize graphs")
parser.add_argument("--softmax_temperature", type=float, default=1.5,
                    help="custom softmax temperature (higher temperature means larger entropy in routing decisions; default: 1.5)")
parser.add_argument("--cost_bound", type=float, default=100.0,
                    help="upper bound on delivery cost to verify (default: 100)")

parser.add_argument("--pretrain_num_episodes", type=int, default=10000,
                    help="pretrain_num_episodes (default: 10000)")

args = parser.parse_args()


os.environ["IGOR_OVERRDIDDED_SOFTMAX_TEMPERATURE"] = str(args.softmax_temperature)

# 1. load scenario
scenario = args.config_file
print(f"Scenario: {scenario}")

with open(scenario) as file:
    scenario_loaded = yaml.load(file, Loader=yaml.FullLoader)

emb_dim = scenario_loaded["settings"]["router"]["dqn_emb"]["embedding"]["dim"]

# graphs size = #sources + #diverters + #sinks + #(conveyors leading to other conveyors)
lengths = [len(scenario_loaded["configuration"][x]) for x in ["sources", "diverters", "sinks"]] \
    + [len([c for c in scenario_loaded["configuration"]["conveyors"].values()
           if c["upstream"]["type"] == "conveyor"])]
#print(lengths)
graph_size = sum(lengths)
print(f"Embedding dimension: {emb_dim}, graph size: {graph_size}")


# 2. pretrain

def pretrain(args, dir_with_models: str, pretrain_filename: str):
    """ ALMOST COPIED FROM THE PRETRAINING NOTEBOOK """
    
    def gen_episodes_progress(num_episodes, **kwargs):
        with tqdm(total=num_episodes) as bar:
            return gen_episodes(bar=bar, num_episodes=num_episodes, **kwargs)
    
    class CachedEmbedding(Embedding):
        def __init__(self, InnerEmbedding, dim, **kwargs):
            self.dim = dim
            self.InnerEmbedding = InnerEmbedding
            self.inner_kwargs = kwargs
            self.fit_embeddings = {}

        def fit(self, graph, **kwargs):
            h = hash_graph(graph)
            if h not in self.fit_embeddings:
                embed = self.InnerEmbedding(dim=self.dim, **self.inner_kwargs)
                embed.fit(graph, **kwargs)
                self.fit_embeddings[h] = embed

        def transform(self, graph, idx):
            h = hash_graph(graph)
            return self.fit_embeddings[h].transform(idx)
    
    def shuffle(df):
        return df.reindex(np.random.permutation(df.index))
    
    def hash_graph(graph):
        if type(graph) != np.ndarray:
            graph = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes))
        m = hashlib.sha256()
        m.update(graph.tobytes())
        return base64.b64encode(m.digest()).decode('utf-8')
    
    def add_inp_cols(tag, dim):
        return mk_num_list(tag + '_', dim) if dim > 1 else tag

    def qnetwork_batches(net, data, batch_size=64, embedding=None):
        n = net.graph_size
        data_cols = []
        amatrix_cols = get_amatrix_cols(n)
        for tag, dim in net.add_inputs:
            data_cols.append(amatrix_cols if tag == 'amatrix' else add_inp_cols(tag, dim))
        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch['addr'].values
            dst = batch['dst'].values
            nbr = batch['neighbour'].values
            if embedding is not None:
                amatrices = batch[amatrix_cols].values
                new_btch = []
                for addr_, dst_, nbr_, A in zip(addr, dst, nbr, amatrices):
                    A = A.reshape(n, n)
                    embedding.fit(A)
                    new_addr = embedding.transform(A, int(addr_))
                    new_dst = embedding.transform(A, int(dst_))
                    new_nbr = embedding.transform(A, int(nbr_))
                    new_btch.append((new_addr, new_dst, new_nbr))
                [addr, dst, nbr] = stack_batch(new_btch)
            addr_inp = torch.tensor(addr, dtype=torch.float)
            dst_inp = torch.tensor(dst, dtype=torch.float)
            nbr_inp = torch.tensor(nbr, dtype=torch.float)
            inputs = tuple(torch.tensor(batch[cols].values, dtype=torch.float) for cols in data_cols)
            output = torch.tensor(batch['predict'].values, dtype=torch.float)
            yield (addr_inp, dst_inp, nbr_inp) + inputs, output

    def qnetwork_pretrain_epoch(net, optimizer, data, **kwargs):
        loss_func = nn.MSELoss()
        for batch, target in qnetwork_batches(net, data, **kwargs):
            optimizer.zero_grad()
            output = net(*batch)
            loss = loss_func(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            yield float(loss)

    def qnetwork_pretrain(net, data, optimizer='rmsprop', epochs=1, save_net=True, **kwargs):
        optimizer = get_optimizer(optimizer)(net.parameters())
        epochs_losses = []
        for i in tqdm(range(epochs)):
            sum_loss = 0
            loss_cnt = 0
            for loss in tqdm(qnetwork_pretrain_epoch(net, optimizer, data, **kwargs), desc=f'epoch {i}'):
                sum_loss += loss
                loss_cnt += 1
            epochs_losses.append(sum_loss / loss_cnt)
        if save_net:
            # label changed by Igor:
            net._label = pretrain_filename
            net.save()
        return epochs_losses
    
    data_conv = gen_episodes_progress(ignore_saved=True, context='conveyors',
                                      num_episodes=args.pretrain_num_episodes,
                                      random_seed=args.random_seed, run_params=scenario,
                                      save_path='../logs/data_conveyor1_oneinp_new.csv')
    data_conv.loc[:, 'working'] = 1.0
    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)
    args = {'scope': dir_with_models, 'activation': 'relu', 'layers': [64, 64], 'embedding_dim': conv_emb.dim}
    conveyor_network_ng_emb = QNetwork(graph_size, **args)
    conveyor_network_ng_emb_ws = QNetwork(graph_size, additional_inputs=[{'tag': 'working', 'dim': 1}], **args)
    conveyor_network_ng_emb_losses = qnetwork_pretrain(conveyor_network_ng_emb, shuffle(data_conv), epochs=10,
                                                       embedding=conv_emb, save_net=True)
    #conveyor_network_ng_emb_ws_losses = qnetwork_pretrain(conveyor_network_ng_emb_ws, shuffle(data_conv), epochs=20,
    #                                                      embedding=conv_emb, save_net=False)

dir_with_models = 'conveyor_test_ng'
filename_suffix = f"_{emb_dim}_{graph_size}_{os.path.split(scenario)[1]}.bin"
pretrain_filename = f"igor_pretrained{filename_suffix}"
pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / pretrain_filename
if args.force_pretrain or not pretrain_path.exists():
    print(f"Pretraining {pretrain_path}...")
    pretrain(args, dir_with_models, pretrain_filename)
else:
    print(f"Using the already pretrained model {pretrain_path}...")


# 3. train

def run_single(file: str, router_type: str, random_seed: int, **kwargs):
    job_id = mk_job_id(router_type, random_seed)
    with tqdm(desc=job_id) as bar:
        queue = DummyProgressbarQueue(bar)
        runner = ConveyorsRunner(run_params=file, router_type=router_type, random_seed=random_seed,
                                 progress_queue=queue, **kwargs)
        event_series = runner.run(**kwargs)
    return event_series, runner

def train(args, dir_with_models: str, pretrain_filename: str, train_filename: str,
          router_type: str, retrain: bool, work_with_files: bool):
    # Igor: I did not see an easy way to change the code in a clean way
    os.environ["IGOR_OVERRIDDEN_DQN_LOAD_FILENAME"] = pretrain_filename
    os.environ["IGOR_TRAIN_PROBABILITY_SMOOTHING"] = str(args.probability_smoothing)
    
    if retrain:
        if "IGOR_OMIT_TRAINING" in os.environ:
            del os.environ["IGOR_OMIT_TRAINING"]
    else:
        os.environ["IGOR_OMIT_TRAINING"] = "True"
    
    event_series, runner = run_single(file=scenario, router_type=router_type, progress_step=500,
                                      ignore_saved=[True], random_seed=args.random_seed)
    
    if router_type == "dqn_emb":
        world = runner.world
        net = next(iter(next(iter(world.handlers.values())).routers.values())).brain
        net._label = train_filename    
        # save or load the trained network
        if work_with_files:
            if retrain:
                net.save()
            else:
                net.restore()
    else:
        world = None
    
    return event_series, world
    

train_filename = f"igor_trained{filename_suffix}"
train_path = Path(TORCH_MODELS_DIR) / dir_with_models / train_filename
retrain = args.force_train or not train_path.exists()
if retrain:
    print(f"Training {train_path}...")
else:
    print(f"Using the already trained model {train_path}...")
_, world = train(args, dir_with_models, pretrain_filename, train_filename, "dqn_emb", retrain, True)


# 4. load the router graph
g = RouterGraph(world)
print("Reachability matrix:")
g.print_reachability_matrix()

def visualize(g: RouterGraph):
    gv_graph = g.to_graphviz()
    prefix = f"../img/topology_graph{filename_suffix}."
    gv_graph.write(prefix + "gv")
    for prog in ["dot", "circo", "twopi"]:
        prog_prefix = f"{prefix}{prog}."
        for fmt in ["pdf", "png"]:
            path = f"{prog_prefix}{fmt}"
            print(f"Drawing {path} ...")
            gv_graph.draw(path, prog=prog, args="-Gdpi=300 -Gmargin=0 -Grankdir=LR")

if not args.skip_graphviz:
    visualize(g)

def smooth(p):
    # smoothing to get rid of 0 and 1 probabilities that lead to saturated gradients
    return (1 - args.probability_smoothing) * p  + args.probability_smoothing / 2

def q_values_to_first_probability(qs: torch.tensor) -> torch.tensor:
    return smooth((qs / args.softmax_temperature).softmax(dim=0)[0])


print(f"Running command {args.command}...")
if args.command == "deterministic_test":
    for source in g.sources:
        for sink in g.sinks:
            print(f"Testing delivery from {source} to {sink}...")
            current_node = source
            visited_nodes = set()
            sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
            while True:
                if current_node in visited_nodes:
                    print("    FAIL due to cycle")
                    break
                visited_nodes.add(current_node)
                print("    in:", current_node)
                if current_node[0] == "sink":
                    print("    ", end="")
                    print("OK" if current_node == sink else "FAIL due to wrong destination")
                    break
                elif current_node[0] in ["source", "junction"]:
                    out_nodes = g.get_out_nodes(current_node)
                    assert len(out_nodes) == 1
                    current_node = out_nodes[0]
                elif current_node[0] == "diverter":
                    current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(current_node, sink)
                    q_values = []
                    for neighbor, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                        with torch.no_grad():
                            q = g.q_forward(current_embedding, sink_embedding, neighbor_embedding).item()
                        print(f"        Q({current_node} -> {neighbor} | {sink}) = {q:.4f}")
                        q_values += [q]
                    best_neighbor_index = np.argmax(np.array(q_values))
                    current_node = neighbors[best_neighbor_index]
                else:
                    raise AssertionError()
elif args.command == "embedding_adversarial":
    adv = PGDAdversary(rho=1.5, steps=100, step_size=0.02, random_start=True, stop_loss=1e5, verbose=2,
                       norm="scaled_l_2", n_repeat=2, repeat_mode="min", dtype=torch.float64)
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        embedding_size = sink_embedding.flatten().shape[0]
        # gather all embeddings that we need to compute the objective
        stored_embeddings = OrderedDict({sink: sink_embedding})
        for node_key in ma.reachable_nodes:
            stored_embeddings[node_key], _, _ = g.node_to_embeddings(node_key, sink)

        def pack_embeddings(embedding_dict: OrderedDict) -> torch.tensor:
            return torch.cat(tuple(embedding_dict.values())).flatten()

        def unpack_embeddings(embedding_vector: torch.tensor) -> OrderedDict:
            embedding_dict = OrderedDict()
            for i, (key, value) in enumerate(stored_embeddings.items()):
                embedding_dict[key] = embedding_vector[i*embedding_size:(i + 1)*embedding_size].reshape(1, embedding_size)
            return embedding_dict

        initial_vector = pack_embeddings(stored_embeddings)

        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            symbolic_objective, objective = ma.get_objective(source)

            def get_gradient(x: torch.tensor) -> Tuple[torch.tensor, float, str]:
                """
                :param x: parameter vector (the one expected to converge to an adversarial example)
                Returns a tuple (gradient pointing to the direction of the adversarial attack,
                                 the corresponding loss function value,
                                 auxiliary information for printing during optimization)."""
                x = Util.optimizable_clone(x.flatten())
                embedding_dict = unpack_embeddings(x)
                objective_inputs = []
                perturbed_sink_embeddings = embedding_dict[sink].repeat(2, 1)
                for diverter in ma.nontrivial_diverters:
                    perturbed_diverter_embeddings = embedding_dict[diverter].repeat(2, 1)
                    _, current_neighbors, _ = g.node_to_embeddings(diverter, sink)
                    perturbed_neighbor_embeddings = torch.cat([embedding_dict[current_neighbor]
                                                               for current_neighbor in current_neighbors])
                    q_values = g.q_forward(perturbed_diverter_embeddings, perturbed_sink_embeddings,
                                           perturbed_neighbor_embeddings).flatten()
                    objective_inputs += [q_values_to_first_probability(q_values)]
                objective_value = objective(*objective_inputs)
                #print(objective_value.detach().cpu().numpy())
                objective_value.backward()
                aux_info = ", ".join([f"{param}={value.detach().cpu().item():.4f}"
                                      for param, value in zip(ma.params, objective_inputs)])
                return x.grad, objective_value.item(), f"[{aux_info}]"
            adv.perturb(initial_vector, get_gradient)
elif args.command == "q_adversarial":
    plot_index = 0
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        embedding_size = sink_embedding.flatten().shape[0]
        sink_embeddings = sink_embedding.repeat(2, 1)

        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            symbolic_objective, objective = ma.get_objective(source)

            # stage 1: linear change of parameters
            filename = "saved-net.bin"
            torch.save(g.q_network.ff_net, filename)
            Util.set_param_requires_grad(g.q_network.ff_net, True)

            def get_objective():
                objective_inputs = []
                for diverter in ma.nontrivial_diverters:
                    diverter_embedding, current_neighbors, neighbor_embeddings = g.node_to_embeddings(diverter, sink)
                    diverter_embeddings = diverter_embedding.repeat(2, 1)
                    neighbor_embeddings = torch.cat(neighbor_embeddings, dim=0)
                    q_values = g.q_forward(diverter_embeddings, sink_embeddings, neighbor_embeddings).flatten()
                    objective_inputs += [q_values_to_first_probability(q_values)]
                return objective(*objective_inputs).item()

            for node_key in g.node_keys:
                current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)

                for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                    with torch.no_grad():
                        reference_q = g.q_forward(current_embedding, sink_embedding, neighbor_embedding).flatten().item()
                    actual_qs = reference_q + np.linspace(-50, 50, 351)

                    opt = torch.optim.SGD(g.q_network.parameters(), lr=0.01)
                    opt.zero_grad()
                    predicted_q = g.q_forward(curent_embedding, sink_embedding, neighbor_embedding).flatten()
                    predicted_q.backward()

                    def gd_step(predicted_q, actual_q, lr: float):
                        for p in g.q_network.parameters():
                            if p.grad is not None:
                                mse_gradient = 2 * (predicted_q - actual_q) * p.grad 
                                p -= lr * mse_gradient
                    #opt = torch.optim.RMSprop(g.q_network.ff_net.parameters(), lr=0.001)
                    objective_values = []
                    lr = 0.001
                    with torch.no_grad():
                        for actual_q in actual_qs:
                            gd_step(predicted_q, actual_q, lr)
                            objective_values += [get_objective()]
                            gd_step(predicted_q, actual_q, -lr)
                    objective_values = torch.tensor(objective_values)
                    label = f"Delivery cost from {source} to {sink} when making optimization step with current={node_key}, neighbor={neighbor_key}"
                    print(f"{label}...")
                    plt.figure(figsize=(14, 2))
                    plt.yscale("log")
                    plt.title(label)
                    plt.plot(actual_qs, objective_values)
                    gap = max(objective_values) - min(objective_values)
                    y_delta = 0 if gap > 0 else 5
                    plt.vlines(reference_q, min(objective_values) - y_delta, max(objective_values) + y_delta)
                    plt.hlines(objective_values[len(objective_values) // 2], min(actual_qs), max(actual_qs))
                    plt.savefig(f"../img/{filename_suffix}_{plot_index}.pdf")
                    plt.close()
                    plot_index += 1
elif args.command == "q_adversarial_lipschitz":
    net = g.q_network.ff_net
    print(net)
    to_sympy = lambda x: sympy.Matrix(x.detach().cpu().numpy())
    A = to_sympy(net.fc1.weight)
    b = to_sympy(net.fc1.bias)
    C = to_sympy(net.fc2.weight)
    d = to_sympy(net.fc2.bias)
    E = to_sympy(net.output.weight)
    f = to_sympy(net.output.bias)
    
    def round_expr(expr: sympy.Expr, num_digits: int) -> sympy.Expr:
        return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})

    def expr_to_string(expr: sympy.Expr) -> str:
        return str(round_expr(expr, 2)).replace("Max(0, ", "ReLU(")
    
    class relu(sympy.Function):
        @classmethod
        def eval(cls, x):
            return x.applyfunc(lambda elem: sympy.Max(elem, 0))

        def _eval_is_real(self):
            return True
    
    def base_bound(expr) -> float:
        if type(expr) in [sympy.Float, sympy.Integer]:
            return np.abs(float(expr))
        if type(expr) == sympy.numbers.NegativeOne:
            return 1.0
        raise RuntimeError(f"Unexpected type {type(expr)} of expression {expr}")
    
    def to_intervals(points):
        return list(zip(points, points[1:]))
    
    def get_subs_value(interval: Tuple[float, float]) -> float:
        return (interval[1] - interval[0]) / 2
    
    def get_bottom_decision_points(expr: sympy.Expr, param: sympy.Symbol) -> Tuple[set, bool]:
        result_set = set()
        all_args_plain = True
        for arg in expr.args:
            arg_set, arg_plain = get_bottom_decision_points(arg, param)
            result_set.update(arg_set)
            all_args_plain = all_args_plain and arg_plain
        if all_args_plain and type(expr) in [sympy.Heaviside, sympy.Max]:
            # new decision point
            index = 1 if type(expr) == sympy.Max else 0
            #solutions = [0]
            #print(expr.args[index])
            # even though the expressions are simple, this works very slowly:
            solutions = sympy.solve(expr.args[index], param)
            #assert len(solutions) == 1
            result_set.add(solutions[0])
            #print(expr, solutions[0])
            all_args_plain = False
        return result_set, all_args_plain
    
    def resolve_bottom_decisions(expr: sympy.Expr, param: sympy.Symbol, param_point: float) -> Tuple[sympy.Expr, bool]:
        if type(expr) in [sympy.Float, sympy.Integer, sympy.numbers.NegativeOne, sympy.Symbol]:
            return expr, True
        if type(expr) in [sympy.Heaviside, sympy.Max]:
            return expr.subs(param, param_point).simplify(), False
        all_args_plain = True
        arg_expressions = []
        for arg in expr.args:
            arg_expr, arg_plain = resolve_bottom_decisions(arg, param, param_point)
            arg_expressions += [arg_expr]
            all_args_plain = all_args_plain and arg_plain
        #print(type(expr), arg_expressions)
        return type(expr)(*arg_expressions), all_args_plain
    
    def estimate_upper_bound(expr: sympy.Expr, param: sympy.Symbol, abs_param_bound: float) -> float:
        points = [p for p in get_bottom_decision_points(expr, param)[0] if np.abs(p) < abs_param_bound]
        points = sorted(points)
        points = [-abs_param_bound] + points + [abs_param_bound]
        print(f"      {points}")
        intervals = to_intervals(points)
        print(f"      {intervals}")
        all_values = set()
        
        for interval in intervals:
            print(f"      {interval}")
            e = resolve_bottom_decisions(expr, param, get_subs_value(interval))[0].simplify()
            final_decision_points = get_bottom_decision_points(e, param)[0]
            final_decision_points = [p for p in final_decision_points if interval[0] < p < interval[1]]
            final_decision_points = [interval[0]] + final_decision_points + [interval[1]]
            refined_intervals = to_intervals(final_decision_points)
            for refined_interval in refined_intervals:
                refined_e = resolve_bottom_decisions(e, param, get_subs_value(refined_interval))[0].simplify()
                derivative = refined_e.diff(param)
                # find stationary points of the derivative
                additional_points = [p for p in sympy.solve(derivative, param)
                                     if refined_interval[0] < p < refined_interval[1]]
                all_points = [refined_interval[0]] + additional_points + [refined_interval[1]]
                all_values.update([np.abs(float(refined_e.subs(param, p).simplify())) for p in all_points])
                print(f"        {refined_interval}")
                print(f"          {expr_to_string(refined_e)}; additional points: {additional_points}")
        return max(all_values)
        
    
    #def estimate_upper_bound(expr, param_name, abs_param_bound) -> float:
    #    if type(expr) == sympy.Add:
    #        return sum([estimate_upper_bound(x, param_name, abs_param_bound) for x in expr.args])
    #    if type(expr) == sympy.Mul:
    #        return np.prod([estimate_upper_bound(x, param_name, abs_param_bound) for x in expr.args])
    #    if type(expr) == sympy.Symbol:
    #        if str(expr) == param_name:
    #            return abs_param_bound
    #        raise RuntimeError(f"Unexpected symbol {expr}")
    #    if type(expr) == sympy.Max:
    #        if len(expr.args) == 2 and str(expr.args[0] == "0"):
    #            return estimate_upper_bound(expr.args[1], param_name, abs_param_bound)
    #        raise RuntimeError(f"Unexpected Max expression {expr}")
    #    if type(expr) == sympy.Heaviside:
    #        return 1.0
    #    return base_bound(expr)
    
    def estimate_top_level_upper_bound(expr, ps_function_names: list, derivative_bounds: dict) -> float:
        if type(expr) == sympy.Add:
            return sum([estimate_top_level_upper_bound(x, ps_function_names, derivative_bounds)
                        for x in expr.args])
        if type(expr) == sympy.Mul:
            return np.prod([estimate_top_level_upper_bound(x, ps_function_names, derivative_bounds)
                            for x in expr.args])
        if type(expr).__name__ in ps_function_names:
            # p(beta) = (1 - smoothing_alpha) sigmoid(logit(beta)) + smoothing_alpha/2
            return 1 - args.probability_smoothing / 2
            #return plain_bounds[type(expr).__name__]
        if type(expr) == sympy.Derivative:
            assert type(expr.args[0]).__name__ in ps_function_names
            # p(beta) = (1 - smoothing_alpha) sigmoid(logit(beta)) + smoothing_alpha/2
            # p'(beta) = (1 - smoothing_alpha) sigmoid'(logit(beta)) logit'(beta)
            # the derivative of sigmoid is at most 1/4
            return derivative_bounds[type(expr.args[0]).__name__] / 4 * (1 - args.probability_smoothing)
        return base_bound(expr)
    
    def to_scalar(x):
        assert x.shape == (1, 1)
        return x[0, 0]
        
    def round_expr(expr: sympy.Expr, num_digits) -> sympy.Expr:
        return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})
        
    beta = sympy.Symbol("β", real=True)
    
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        embedding_size = sink_embedding.flatten().shape[0]
        sink_embeddings = sink_embedding.repeat(2, 1)
        
        lr = 0.001
        delta_q_max = 10.0
        # 2 comes from differentiating a square
        beta_bound = 2 * lr * delta_q_max
        
        ps_function_names = [f"p{i}" for i in range(len(ma.params))]
        function_ps = [sympy.Function(name) for name in ps_function_names]
        evaluated_function_ps = [f(beta) for f in function_ps]
        
        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            symbolic_objective, objective = ma.get_objective(source)
        
            for node_key in g.node_keys:
                current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)

                for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                    print(f"    Considering learning step {node_key} -> {neighbor_key}...")
                    opt = torch.optim.SGD(g.q_network.parameters(), lr=0.01)
                    opt.zero_grad()
                    predicted_q = g.q_forward(current_embedding, sink_embedding, neighbor_embedding).flatten()
                    predicted_q.backward()
                    print(f"      Reference Q value = {predicted_q.item()}")

                    A_hat = to_sympy(net.fc1.weight.grad)
                    b_hat = to_sympy(net.fc1.bias.grad)
                    C_hat = to_sympy(net.fc2.weight.grad)
                    d_hat = to_sympy(net.fc2.bias.grad)
                    E_hat = to_sympy(net.output.weight.grad)
                    f_hat = to_sympy(net.output.bias.grad)
                    
                    MOCK = True
                    if MOCK:
                        dim = 6
                        #print(A.shape, b.shape, C.shape, d.shape, E.shape, f.shape)
                        A = A[:dim, :]; A_hat = A_hat[:dim, :]
                        b = b[:dim, :]; b_hat = b_hat[:dim, :]
                        C = C[:dim, :dim]; C_hat = C_hat[:dim, :dim]
                        d = b[:dim, :]; d_hat = d_hat[:dim, :]
                        E = E[:, :dim]; E_hat = E_hat[:, :dim]

                    def sympy_q(beta, x):
                        result = relu((A + beta * A_hat) @ x      + b + beta * b_hat)
                        result = relu((C + beta * C_hat) @ result + d + beta * d_hat)
                        return        (E + beta * E_hat) @ result + f + beta * f_hat
                    
                    print(f"      cost(p) = {symbolic_objective}")
                    assert type(symbolic_objective) == sympy.Mul
                    
                    # walk through the product
                    # if this is pow(something, -1), something is the denominator
                    # the rest goes to the numerator
                    
                    nominator = 1
                    
                    for arg in symbolic_objective.args:
                        if type(arg) == sympy.Pow:
                            assert type(arg.args[1]) == sympy.numbers.NegativeOne, type(arg.args[1])
                            denominator = arg.args[0]
                        else:
                            nominator *= arg 
                    
                    print(f"      nominator(p) = {nominator}")
                    print(f"      denominator(p) = {denominator}")
                    
                    # compute the sign of v, then ensure that it is "+"
                    # the values to subsitute are arbitrary within (0, 1)
                    denominator_value = denominator.subs([(param, 0.5) for param in ma.params]).simplify()
                    print(f"      denominator(0.5) = {denominator_value:.4f}")
                    if float(str(denominator_value)) < 0:
                        nominator *= -1
                        denominator *= -1
                    kappa = nominator - args.cost_bound * denominator
                    print(f"      κ(p) = {kappa}, κ(p) < 0?")
                    
                    kappa = kappa.subs(list(zip(ma.params, evaluated_function_ps)))
                    print(f"      κ(β) = {kappa}, κ(β) < 0?")
                    dkappa_dbeta = kappa.diff(beta)
                    print(f"      dκ(β)/dβ = {dkappa_dbeta}")
                    
                    #  compute a pool of bounds
                    plain_bounds = {}
                    derivative_bounds = {}
                    for param, diverter_key in zip(ma.params, ma.nontrivial_diverters):
                        diverter_embedding, current_neighbors, neighbor_embeddings = g.node_to_embeddings(diverter_key, sink)
                        print(f"      Computing logits for {param} = P{diverter_key} -> {current_neighbors[0]})....")
                        #print(param, diverter_key)

                        delta_e = lambda nbr: to_sympy(torch.cat((diverter_embedding - sink_embedding,
                                                                  nbr - sink_embedding), dim=1).T)
                        delta_e1 = delta_e(neighbor_embeddings[0])
                        delta_e2 = delta_e(neighbor_embeddings[1])
                        #print(delta_e1, delta_e2)
                        
                        logit = to_scalar(sympy_q(beta, delta_e1) - sympy_q(beta, delta_e2))
                        print(f"      logit = {str(round_expr(logit, 3))[:500]} ...")
                        dlogit_dbeta = logit.diff(beta)
                        print(f"      dlogit_dbeta = {str(round_expr(dlogit_dbeta, 3))[:500]} ...")
                        print(f"      Computing logit bounds for {param} = P{diverter_key} -> {current_neighbors[0]})...")
                        derivative_bounds[param.name] = estimate_upper_bound(dlogit_dbeta, beta, beta_bound)

                    print(f"      Computing the final upper bound on dκ(β)/dβ...")
                    top_level_bound = estimate_top_level_upper_bound(dkappa_dbeta, ps_function_names, derivative_bounds)
                    print(f"      Final upper bound on the Lipschitz constant = {top_level_bound}")
                    
                    # TODO prove an upper bound on the function for each interval
    
elif args.command == "compare":
    _legend_txt_replace = {
        'networks': {
            'link_state': 'Shortest paths', 'simple_q': 'Q-routing', 'pred_q': 'PQ-routing',
            'glob_dyn': 'Global-dynamic', 'dqn': 'DQN', 'dqn_oneout': 'DQN (1-out)',
            'dqn_emb': 'DQN-LE', 'centralized_simple': 'Centralized control'
        }, 'conveyors': {
            'link_state': 'Vyatkin-Black', 'simple_q': 'Q-routing', 'pred_q': 'PQ-routing',
            'glob_dyn': 'Global-dynamic', 'dqn': 'DQN', 'dqn_oneout': 'DQN (1-out)',
            'dqn_emb': 'DQN-LE', 'centralized_simple': 'BSR'
        }
    }
    _targets = {'time': 'avg', 'energy': 'sum', 'collisions': 'sum'}
    _ylabels = {
        'time': 'Mean delivery time', 'energy': 'Total energy consumption', 'collisions': 'Cargo collisions'
    }
    
    router_types = ["dqn_emb", "link_state", "simple_q"]
    series = []
    for router_type in router_types:
        s, _ = train(args, dir_with_models, pretrain_filename, train_filename, router_type, True, False)
        series += [s.getSeries(add_avg=True)]
    
    dfs = []
    for router_type, s in zip(router_types, series):
        df = s.copy()
        add_cols(df, router_type=router_type, seed=args.random_seed)
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    
    def print_sums(df):
        types = set(df['router_type'])
        for tp in types:
            x = df.loc[df['router_type']==tp, 'count'].sum()
            txt = _legend_txt_replace.get(tp, tp)
            print('  {}: {}'.format(txt, x))
    
    def plot_data(data, meaning='time', figsize=(15,5), xlim=None, ylim=None,
              xlabel='Simulation time', ylabel=None, font_size=14, title=None, save_path=None,
              draw_collisions=False, context='networks', **kwargs):
        if 'time' not in data.columns:
            datas = split_dataframe(data, preserved_cols=['router_type', 'seed'])
            for tag, df in datas:
                if tag == 'collisions' and not draw_collisions:
                    print('Number of collisions:')
                    print_sums(df)
                    continue

                xlim = kwargs.get(tag+'_xlim', xlim)
                ylim = kwargs.get(tag+'_ylim', ylim)
                save_path = kwargs.get(tag+'_save_path', save_path)
                plot_data(df, meaning=tag, figsize=figsize, xlim=xlim, ylim=ylim,
                          xlabel=xlabel, ylabel=ylabel, font_size=font_size,
                          title=title, save_path=save_path, context='conveyors')
            return 

        target = _targets[meaning]
        if ylabel is None:
            ylabel = _ylabels[meaning]

        fig = plt.figure(figsize=figsize)
        ax = sns.lineplot(x='time', y=target, hue='router_type', data=data, err_kws={'alpha': 0.1})

        handles, labels = ax.get_legend_handles_labels()
        new_labels = list(map(lambda l: _legend_txt_replace[context].get(l, l), labels[1:]))
        ax.legend(handles=handles[1:], labels=new_labels, fontsize=font_size)

        ax.tick_params(axis='both', which='both', labelsize=int(font_size*0.75))

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if save_path is not None:
            fig.savefig('../img/' + save_path, bbox_inches='tight')
    
    plot_data(dfs, figsize=(10, 8), font_size=22, energy_ylim=(7e6, 2.3e7),
              time_save_path='conveyors-break-1-time.pdf', energy_save_path='conveyors-break-1-energy.pdf')

else:
    raise RuntimeError(f"Unknown command {args.command}.")