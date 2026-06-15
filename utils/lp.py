import math
from ortools.sat.python import cp_model

def solve_episode_mip(df_episode, config):
    """
    Solves the offline optimal scheduling problem for a steel slab yard using Google OR-Tools (CP-SAT).
    
    This Mixed-Integer Programming (MIP) model calculates the theoretical maximum utility for a given sequence of 
    slab arrivals by optimizing their routing through thermal isolation rooms (Fixed/Mobile) vs the open yard.
    
    Args:
        df_episode (pd.DataFrame): A DataFrame containing the slab arrival history (id, step/arrival_time, 
                                   initial_wait_time, initial_temp) for a single episode.
        config (module): The configuration module containing physical parameters, capacities, costs, and limits.
        
    Returns:
        dict: A dictionary containing:
            - 'status' (str): The solver's completion state ('Optimal', 'Feasible', or 'Infeasible').
            - 'objective' (float): The maximum calculated theoretical utility score.
            - 'failed_slabs' (int): The total count of slabs delivered below the temperature threshold.
              Returns None if no slabs exist in the episode.
    """
    model = cp_model.CpModel()
    
    # Single uniform scale factor for all floating-point numbers
    SCALE = 1000  # 3 decimal places of precision
    
    # Parse Config
    temp_dict = config.temp_dict
    ambient = [temp_dict['yard'][0], temp_dict['fixed'][0], temp_dict['mobile'][0]]
    k = [temp_dict['yard'][1], temp_dict['fixed'][1], temp_dict['mobile'][1]]
    
    # Scale crane budgets & costs
    crane_move_time = config.crane_move_time
    crane_time_int = [int(round(t * SCALE)) for t in crane_move_time]
    
    crane_costs = [t * config.crane_cost_per_time_step for t in crane_move_time]
    crane_cost_int = [int(round(c * SCALE)) for c in crane_costs]
    
    cap_fixed = config.nhotroom_fixed
    cap_mobile = config.nhotroom_mobile
    
    # Scale thresholds & penalties
    threshold_scaled = int(round(config.temp_threshold * SCALE))
    penalty_int = int(round(config.temp_treshold_utility_cost * SCALE))
    
    # Calculate integer decay coefficients
    # Formula: temp(t) = ambient + (temp(t-1) - ambient) * exp(-k)
    # We factor this to: temp(t) = temp(t-1)*exp(-k) + ambient*(1 - exp(-k))
    decay_int = [int(round(math.exp(-k[j]) * SCALE)) for j in range(3)]
    const_int = [int(round(ambient[j] * SCALE * (1.0 - math.exp(-k[j])))) for j in range(3)]
    
    # 1. Input Data
    slabs = []
    for _, row in df_episode.iterrows():
        slabs.append({
            'id': int(row['id']),
            'arrival_time': int(row['step']),
            'delivery_time': int(row['step'] + row['initial_wait_time']),
            'initial_temp': float(row['initial_temp'])
        })
        
    if not slabs:
        return None
        
    max_t = max(s['delivery_time'] for s in slabs)
    min_t = min(s['arrival_time'] for s in slabs)
    
    # 3. Decision Variables
    x = {}
    move = {}
    temp = {}
    is_failed = {}
    
    for s in slabs:
        sid = s['id']
        is_failed[sid] = model.NewBoolVar(f'failed_{sid}')
        for t in range(s['arrival_time'], s['delivery_time']):
            # Maximum theoretical scaled temp: 3000 * 1000 = 3,000,000
            temp[sid, t] = model.NewIntVar(0, 3000 * SCALE, f'temp_{sid}_{t}')
            for j in range(3):
                x[sid, j, t] = model.NewBoolVar(f'x_{sid}_{j}_{t}')
                move[sid, j, t] = model.NewBoolVar(f'move_{sid}_{j}_{t}')
                
    # 4. Logic & Capacity Constraints
    for s in slabs:
        sid = s['id']
        for t in range(s['arrival_time'], s['delivery_time']):
            model.AddExactlyOne(x[sid, j, t] for j in range(3))
            
    for t in range(min_t, max_t):
        active_sids = [s['id'] for s in slabs if s['arrival_time'] <= t < s['delivery_time']]
        if not active_sids:
            continue
        model.Add(sum(x[sid, 1, t] for sid in active_sids) <= cap_fixed)
        model.Add(sum(x[sid, 2, t] for sid in active_sids) <= cap_mobile)
        
        # Hourly Crane Budget (Max 1.0 hr * SCALE)
        model.Add(sum(crane_time_int[j] * move[sid, j, t] for sid in active_sids for j in range(3)) <= 1 * SCALE)
        
    # 5. Define Movement
    for s in slabs:
        sid = s['id']
        arr = s['arrival_time']
        for t in range(arr, s['delivery_time']):
            for j in range(3):
                if t == arr:
                    prev_x = 1 if j == 0 else 0
                    model.Add(move[sid, j, t] >= x[sid, j, t] - prev_x)
                else:
                    model.Add(move[sid, j, t] >= x[sid, j, t] - x[sid, j, t-1])
                    
    # 6. Thermodynamics
    for s in slabs:
        sid = s['id']
        arr = s['arrival_time']
        for t in range(arr, s['delivery_time']):
            for j in range(3):
                if t == arr:
                    init_t_scaled = int(round(s['initial_temp'] * SCALE))
                    # Math: (T_init * decay_int) / SCALE + const_int
                    expr_val = (init_t_scaled * decay_int[j]) // SCALE + const_int[j]
                    model.Add(temp[sid, t] == expr_val).OnlyEnforceIf(x[sid, j, t])
                else:
                    # Intermediate variable to handle multiplication before division
                    temp_j = model.NewIntVar(0, 3000 * SCALE * SCALE, f'temp_j_{sid}_{j}_{t}')
                    
                    expr = temp[sid, t-1] * decay_int[j]
                    
                    # Instead of AddDivisionEquality (which is messy), we use integer division approximation
                    # by bounding temp[sid, t] appropriately. Since CP-SAT doesn't natively divide variables cleanly:
                    
                    model.Add(temp_j == expr + const_int[j] * SCALE)
                    
                    # Approximate temp[sid, t] = temp_j / SCALE
                    model.Add(temp[sid, t] * SCALE <= temp_j).OnlyEnforceIf(x[sid, j, t])
                    model.Add(temp[sid, t] * SCALE >= temp_j - SCALE + 1).OnlyEnforceIf(x[sid, j, t])
                    
    # 7. Quality Penalty
    # BIG_M linearized formulation: temp_final >= threshold - BIG_M * fail
    BIG_M = 1000 * SCALE
    for s in slabs:
        sid = s['id']
        t_final = s['delivery_time'] - 1
        if t_final >= s['arrival_time']:
            model.Add(temp[sid, t_final] >= threshold_scaled - BIG_M * is_failed[sid])
            
    # 8. Objective Function
    objective_terms = []
    for s in slabs:
        sid = s['id']
        t_final = s['delivery_time'] - 1
        if t_final >= s['arrival_time']:
            objective_terms.append(temp[sid, t_final])
            objective_terms.append(-penalty_int * is_failed[sid])
            
        for t in range(s['arrival_time'], s['delivery_time']):
            for j in range(3):
                if crane_cost_int[j] > 0:
                    objective_terms.append(-crane_cost_int[j] * move[sid, j, t])
                    
    model.Maximize(sum(objective_terms))
    
    # 9. Solve
    solver = cp_model.CpSolver()
    
    # Set time limit from config
    time_limit_seconds = getattr(config, 'mip_time_limit_seconds', 60)
    solver.parameters.max_time_in_seconds = time_limit_seconds
    
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return {
            'status': 'Optimal' if status == cp_model.OPTIMAL else 'Feasible',
            'objective': solver.ObjectiveValue() / SCALE,
            'failed_slabs': sum(solver.Value(is_failed[s['id']]) for s in slabs)
        }
    else:
        return {
            'status': 'Infeasible',
            'objective': None,
            'failed_slabs': None
        }