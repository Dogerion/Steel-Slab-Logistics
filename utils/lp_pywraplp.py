import math
from ortools.linear_solver import pywraplp

def solve_episode_mip(df_episode, config):
    # Create the linear solver with the CP-SAT backend.
    solver = pywraplp.Solver.CreateSolver('SAT')
    if not solver:
        return None

    BIG_M = 1000.0
    
    # Parse Config for Physical Constants
    temp_dict = config.temp_dict
    ambient = [temp_dict['yard'][0], temp_dict['fixed'][0], temp_dict['mobile'][0]]
    k = [temp_dict['yard'][1], temp_dict['fixed'][1], temp_dict['mobile'][1]]
    crane_move_time = config.crane_move_time
    crane_costs = [t * config.crane_cost_per_time_step for t in crane_move_time]
    cap_fixed = config.nhotroom_fixed
    cap_mobile = config.nhotroom_mobile
    temp_threshold = config.temp_threshold
    penalty_cost = config.temp_treshold_utility_cost
    
    # 1. Input Data Structure (Slabs)
    slabs = []
    for _, row in df_episode.iterrows():
        slabs.append({
            'id': int(row['id']),
            'arrival_time': int(row['step']),
            'delivery_time': int(row['step'] + row['initial_wait_time']),
            'initial_temp': float(row['initial_temp'])
        })
        
    max_t = max([s['delivery_time'] for s in slabs]) if slabs else 0
    min_t = min([s['arrival_time'] for s in slabs]) if slabs else 0
    
    # 3. Decision Variables
    x = {}
    move = {}
    temp = {}
    is_failed = {}
    
    for s in slabs:
        sid = s['id']
        is_failed[sid] = solver.BoolVar(f'failed_{sid}')
        for t in range(s['arrival_time'], s['delivery_time']):
            temp[sid, t] = solver.NumVar(0.0, 1000.0, f'temp_{sid}_{t}')
            for j in range(3):
                x[sid, j, t] = solver.BoolVar(f'x_{sid}_{j}_{t}')
                move[sid, j, t] = solver.BoolVar(f'move_{sid}_{j}_{t}')
                
    # 4. Constraints: Logic and Capacity
    for s in slabs:
        sid = s['id']
        for t in range(s['arrival_time'], s['delivery_time']):
            solver.Add(sum(x[sid, j, t] for j in range(3)) == 1)
            
    for t in range(min_t, max_t):
        active_sids = [s['id'] for s in slabs if s['arrival_time'] <= t < s['delivery_time']]
        if not active_sids:
            continue
        solver.Add(sum(x[sid, 1, t] for sid in active_sids) <= cap_fixed)
        solver.Add(sum(x[sid, 2, t] for sid in active_sids) <= cap_mobile)
        
        # 5. Constraints: Crane Budget (max 1.0 hour per hour)
        solver.Add(sum(crane_move_time[j] * move[sid, j, t] for sid in active_sids for j in range(3)) <= 1.0)
        
    # 5. Define Movement
    for s in slabs:
        sid = s['id']
        arr = s['arrival_time']
        for t in range(arr, s['delivery_time']):
            for j in range(3):
                if t == arr:
                    prev_x = 1 if j == 0 else 0
                    solver.Add(move[sid, j, t] >= x[sid, j, t] - prev_x)
                else:
                    solver.Add(move[sid, j, t] >= x[sid, j, t] - x[sid, j, t-1])
                    
    # 6. Constraints: Thermodynamics
    for s in slabs:
        sid = s['id']
        arr = s['arrival_time']
        for t in range(arr, s['delivery_time']):
            for j in range(3):
                decay_factor = math.exp(-k[j])
                if t == arr:
                    decayed_value = ambient[j] + (s['initial_temp'] - ambient[j]) * decay_factor
                    solver.Add(temp[sid, t] <= decayed_value + BIG_M * (1 - x[sid, j, t]))
                    solver.Add(temp[sid, t] >= decayed_value - BIG_M * (1 - x[sid, j, t]))
                else:
                    const_term = ambient[j] * (1 - decay_factor)
                    solver.Add(temp[sid, t] <= const_term + decay_factor * temp[sid, t-1] + BIG_M * (1 - x[sid, j, t]))
                    solver.Add(temp[sid, t] >= const_term + decay_factor * temp[sid, t-1] - BIG_M * (1 - x[sid, j, t]))

    # 7. Constraints: Quality Penalty
    for s in slabs:
        sid = s['id']
        t_final = s['delivery_time'] - 1
        if t_final >= arr:
            solver.Add(temp[sid, t_final] >= temp_threshold - BIG_M * is_failed[sid])
            
    # 8. Objective Function
    objective = solver.Objective()
    for s in slabs:
        sid = s['id']
        t_final = s['delivery_time'] - 1
        if t_final >= s['arrival_time']:
            objective.SetCoefficient(temp[sid, t_final], 1.0)
            objective.SetCoefficient(is_failed[sid], -penalty_cost)
            
        for t in range(s['arrival_time'], s['delivery_time']):
            for j in range(3):
                objective.SetCoefficient(move[sid, j, t], -crane_costs[j])
                
    objective.SetMaximization()
    
    # Set a time limit for the solver from config to prevent it from getting stuck
    solver.set_time_limit(getattr(config, 'mip_time_limit_ms', 10000))
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        return {
            'status': 'Optimal' if status == pywraplp.Solver.OPTIMAL else 'Feasible',
            'objective': objective.Value(),
            'failed_slabs': sum(is_failed[s['id']].solution_value() for s in slabs)
        }
    else:
        return {
            'status': 'Infeasible',
            'objective': None,
            'failed_slabs': None
        }