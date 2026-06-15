# Mixed-Integer Programming (MIP) Model for Steel Slab Logistics

This document details the mathematical formulation of the theoretical optimum scheduling model implemented using Google OR-Tools (CP-SAT backend) in `utils/lp.py`. The model evaluates the absolute upper bound of performance (the mathematical optimum) for the offline scheduling of simulated steel slabs.

---

## 1. Problem Description
The factory processes hot steel slabs arriving continuously into a storage yard. Slabs must be moved into limited-capacity thermal isolation rooms (Fixed or Mobile Covers) using a crane to slow their temperature decay. The objective is to maximize the final delivered temperature of the slabs minus operational crane costs and thermal penalties of any failed slabs.

---

## 2. Indices & Sets
* $s \in S$: Set of all simulated slabs arriving during the horizon.
* $t \in T$: Set of time steps (hours) from $t_{min}$ to $t_{max}$.
* $j \in \{0, 1, 2\}$: Set of possible locations.
  * $j=0$: Open Yard (Infinite capacity, fast decay, no cost)
  * $j=1$: Fixed Hot Room (Capacity 4, slowest decay, highest crane cost)
  * $j=2$: Mobile Cover (Capacity 2, medium decay, medium crane cost)

---

## 3. Parameters
For each slab $s$:
* $arr_s$: Arrival time (step) into the yard.
* $del_s$: Delivery time (step) when the slab leaves the system.
* $T_{init, s}$: Initial temperature of slab $s$ upon arrival ($^\circ C$).

For each location $j$:
* $A_j$: Ambient temperature of the location $j$.
* $k_j$: Cooling constant at the location $j$.
* $c_j$: Cost to move a slab into location $j$ using the crane.
* $u_j$: Time required (in hours) to move a slab into location $j$.
* $Cap_j$: Maximum number of slabs allowed in location $j$ concurrently.

Global Parameters:
* $T_{thresh}$: Temperature threshold (e.g., $700^\circ C$). If a slab falls below this, a penalty is applied.
* $P$: Penalty cost for failing to meet the temperature threshold.
* $MAX\_CRANE = 1.0$: Maximum total hours the crane can operate per time step.
* $SCALE = 1000$: A single global scale factor used to convert all physical constants, continuous temperatures, and costs into integer for CP-SAT requirement.

---

## 4. Variables
### 4.1 Decision Variables
For all active periods $arr_s \le t < del_s$:
* $x_{s, j, t} \in \{0, 1\}$: 1 if slab $s$ is stored in location $j$ during hour $t$.
* $move_{s, j, t} \in \{0, 1\}$: 1 if slab $s$ entered location $j$ during hour $t$ (utulizing a crane).

### 4.2 Dependent Variables
* $temp_{s, t} \in \mathbb{Z}^+$: The exact scaled temperature of slab $s$ at the *end* of hour $t$.
* $fail_s \in \{0, 1\}$: 1 if slab $s$ drops below the threshold temperature $T_{thresh}$ at the time of delivery.

---

## 5. Constraints

### 5.1. Logic and Exclusivity
A slab must be in exactly one location at any given time it is in the system.
$$ \sum_{j=0}^{2} x_{s, j, t} = 1 \quad \forall s, \forall t \in [arr_s, del_s) $$

### 5.2. Capacity Constraints
The number of slabs in the fixed room and mobile covers cannot exceed their physical limits at any hour $t$.
$$ \sum_{s \text{ active}} x_{s, 1, t} \le Cap_1 \quad \forall t $$
$$ \sum_{s \text{ active}} x_{s, 2, t} \le Cap_2 \quad \forall t $$

*(Note: The yard $j=0$ is assumed to have infinite capacity.)*

### 5.3. Crane Movement & Hourly Budget
A movement is triggered if a slab's location $j$ at hour $t$ is different from its location at $t-1$.
$$ move_{s, j, t} \ge x_{s, j, t} - x_{s, j, t-1} \quad \forall s, \forall j, t > arr_s $$
*(For $t = arr_s$, slabs arrive automatically in the yard ($j=0$) without crane cost: $move_{s, j, t} \ge x_{s, j, t} - (1 \text{ if } j=0 \text{ else } 0)$).*

The crane can only work for a maximum of 1 hour per time step:
$$ \sum_{s \text{ active}} \sum_{j=0}^{2} u_j \cdot move_{s, j, t} \le 1.0 \quad \forall t $$

### 5.4. Thermodynamics (Newton's Law of Cooling)
The temperature of each active slab is updated hour-by-hour according to Newton's Law of Cooling, where the cooling rate and ambient environment are governed by the slab's current location $j$.

For $t = arr_s$ (First Hour):
$$ temp_{s, t} = A_j + (T_{init, s} - A_j) \cdot \exp(-k_j) \quad \text{when } x_{s, j, t} = 1 $$

For $t > arr_s$ (Subsequent Hours):
$$ temp_{s, t} = A_j + (temp_{s, t-1} - A_j) \cdot \exp(-k_j) \quad \text{when } x_{s, j, t} = 1 $$

### 5.5. Quality Penalty
This constraint ensures that if a slab's final delivery temperature falls below the quality threshold $T_{thresh}$, the dependent failure indicator variable $fail_s$ is forced to 1. This captures a product failure and applies the corresponding penalty $P$ in the objective function.
$$ temp_{s, t_{final}} \ge T_{thresh} - M \cdot fail_s \quad \forall s $$

---

## 6. Objective Function
The goal is to maximize the utility of the delivered steel by maximizing final temperatures while penalizing excessive crane operations and temperature failures.

**Maximize:**
$$ Z = \sum_{s} temp_{s, del_s - 1} - \sum_{s} \sum_{t} \sum_{j} (c_j \cdot move_{s, j, t}) - \sum_{s} (P \cdot fail_s) $$

---

## 7. Assumptions & Limitations
1. **Perfect Information:** The MIP solver has perfect knowledge of all future slab arrivals ($arr_s$) and their specific cooling constants over the horizon. It solves the offline theoretical optimum, while the RL agent must operate online without having any information about the future.
2. **Deterministic Physics:** Cooling rates are treated as perfectly deterministic equations within the solver.
3. **Instantaneous Moves:** While crane capacity consumes a budget of the hour, the slab is assumed to spend the entirety of the hour cooling under the thermodynamic profile of its final destination location $j$.
