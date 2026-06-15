# Environment Parameters
urgent_slabs_count = 4

## Slab Parameters
### Temperature Parameters
slab_mean_temp = 900
slab_mean_std = 20
slab_temp_ubound = 1000

### Quantity Parameters
slab_quantity_probability = [0.2, 0.5, 0.3] # Probabilities for selecting each slab type (Must sum to 1)

### Wait Time Parameters
urgency_bounds = [4, 8] # Discrete uniformly distributed wait times for slabs

## Temperature Parameters
temp_dict = {
    "yard": [25, 0.95],
    "fixed": [650, 0.03],
    "mobile": [300, 0.05]
}

# Optimization Parameters

## Count Parameters
ncrane = 1 # Number of cranes available for loading
nhotroom_fixed = 4 # Number of fixed hot rooms available for cooling slabs
nhotroom_mobile = 2 # Number of mobile hot rooms available for cooling slabs

crane_move_time = [0,1/2,3/4] # Crane movement times for storage type

## Cost Parameters
crane_cost_per_time_step = 60 # Cost of operating a crane per time step
celsius_utility_cost = 1 # Cost per Celsius lost

### Temp Treshold for Utility Calculation
temp_threshold = 700 # Temperature threshold for utility calculation (Celsius)
temp_treshold_utility_cost = 500 # Cost if slab temperature falls below threshold

# Reinforcement Learning Parameters
train_time_frame = 100
train_step_count = 2e5
# train_step_count = 40e3
test_time_frame = 12
test_step_count =120

action_space_size = [3 for _ in range(urgent_slabs_count)] # State space dimensions: Action 0 (Leave in Yard), Action 1 (Pull to Fixed Room), Action 2 (Pull to Mobile Cover) for 4 urgent slabs
state_space_size = 11
# Wait time of the 1st most urgent yard slab.
# Wait time of the 2nd most urgent yard slab.
# Wait time of the 3rd most urgent yard slab.
# Wait time of the 4th most urgent yard slab.
# Normalized temperature of the 1st most urgent yard slab.
# Normalized temperature of the 2nd most urgent yard slab.
# Normalized temperature of the 3rd most urgent yard slab.
# Normalized temperature of the 4th most urgent yard slab.
# Available slots in the Fixed Hot Room (0 to 4).
# Available Mobile Covers (0 to 2).
# Total Yard Backlog (Integer count of all slabs currently in the yard).


## MIP Solver Configurations
mip_time_limit_seconds = 60*5 #Set this to sufficiently long time for solver to reach optimal solution

