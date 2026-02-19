"""
Minimal Rebalancing Cost 
------------------------
This file contains the specifications for the Min Reb Cost problem.
"""
from collections import defaultdict
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, CPLEX_PY


def solveRebFlow(env, desiredAcc, agent_id):

    t = env.time
    edges = [(i, j) for i, j in env.G.edges]

    # Map vehicle availability and desired vehicles for each region
    #accTuple = [(n, int(env.agent_acc[agent_id][n][t+1])) for n in env.agent_acc[agent_id]]
    acc_init = {n: int(env.agent_acc[agent_id][n][t+1]) for n in env.agent_acc[agent_id]}
    #acc_init = {n: int(env.acc[n][t+1]) for n in env.acc}
    desired_vehicles = {n: int(round(desiredAcc[n])) for n in desiredAcc}

    # --- Integer verification logging ---
    for n in acc_init:
        raw = env.agent_acc[agent_id][n][t+1]
        if not isinstance(raw, int) or raw != int(raw):
            print(f"[t={t}] agent {agent_id} | acc_init[{n}] raw={raw} (type={type(raw).__name__}), cast to {acc_init[n]}")
    for n in desired_vehicles:
        raw = desiredAcc[n]
        if not isinstance(raw, int) or raw != int(raw):
            print(f"[t={t}] agent {agent_id} | desiredAcc[{n}] raw={raw} (type={type(raw).__name__}), rounded to {desired_vehicles[n]}")

    region = [n for n in acc_init]
    # Time on each edge (used in the objective)
    time = {(i, j): env.G.edges[i, j]['time'] for i, j in edges}
    
    # Log edge times
    non_int_times = {e: time[e] for e in time if time[e] != int(time[e])}
    if non_int_times:
        print(f"[t={t}] agent {agent_id} | Non-integer edge times: {non_int_times}")
    
    print(f"[t={t}] agent {agent_id} | acc_init={acc_init} | desired={desired_vehicles} | sum_acc={sum(acc_init.values())} sum_desired={sum(desired_vehicles.values())}")

    tol = 1e-6
    
    def build_model(var_cat):
        # Define the PuLP problem
        model = LpProblem("RebalancingFlowMinimization", LpMinimize)
        
        # Decision variables: rebalancing flow on each edge
        rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat=var_cat) for (i, j) in edges}

        # Objective: minimize total time (cost) of rebalancing flows
        model += lpSum(rebFlow[(i, j)] * time[(i, j)] for (i, j) in edges), "TotalRebalanceCost"
        
        # Constraints for each region (node)
        for k in region:
            # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
            model += (
                lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
            ) >= desired_vehicles[k] - acc_init[k], f"FlowConservation_{k}"

            # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
            model += (
                lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= acc_init[k], 
                f"RebalanceSupply_{k}"
            )
        return model, rebFlow
    
    model, rebFlow = build_model('Continuous')
    status = model.solve(CPLEX_PY(msg=False))
    if LpStatus[status] != "Optimal":
        print(f"Optimization failed with status: {LpStatus[status]}")
        return None
    else: 
        fractional = False
        flow = defaultdict(float)
        for (i, j) in edges:
            flow[(i, j)] = rebFlow[(i, j)].varValue
            if abs(flow[(i, j)] - round(flow[(i, j)])) > tol: 
                fractional = True
                break 
        if fractional:
            print("Fractional solution detected, re-solving with integer constraints...")
            model, rebFlow = build_model('Integer')
            status = model.solve(CPLEX_PY(msg=False))
            if LpStatus[status] != "Optimal":
                print(f"Optimization failed with status: {LpStatus[status]} when enforcing integer constraints")
                return None
            else:
                flow = defaultdict(float)
                for (i, j) in edges:
                    flow[(i, j)] = rebFlow[(i, j)].varValue
        action = [int(round(flow[i,j])) for i,j in env.edges]
        return action
 
