# TATA: Steel Slab Logistics Optimization

This project models and solves a Steel Slab Logistics Scheduling and Routing problem. The factory processes hot steel slabs arriving continuously into a storage yard. To slow their temperature decay, slabs must be moved into limited-capacity thermal isolation rooms (Fixed or Mobile Covers) using a crane. The goal is to maximize the final delivered temperature of the slabs while minimizing operational crane costs and penalties for slabs dropping below a critical temperature threshold.

The project implements two approaches to solve this problem:
1. **Reinforcement Learning (RL):** An online scheduling agent using Proximal Policy Optimization (PPO) via Stable Baselines3.
2. **Mixed-Integer Programming (MIP):** A theoretical mathematical optimum solver implemented using Google OR-Tools (CP-SAT backend) for the offline version of the problem.

Because the MIP solver operates offline with perfect knowledge of all future slab arrivals, it can compute the absolute mathematical optimum. This serves as a theoretical upper bound and a comparison baseline for evaluating the online RL agent, which must make real-time decisions without any future information.

## Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast and reliable Python package management. 

### 1. Install `uv`
If you haven't installed `uv` yet, you can do so by following the official [installation instructions](https://github.com/astral-sh/uv?tab=readme-ov-file#installation), or simply run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(On Windows, use `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)*

### 2. Install Dependencies
Once `uv` is installed, set up the environment and install the required packages by running:
```bash
uv sync
```
This command will create a virtual environment and install all necessary dependencies specified in the `pyproject.toml` file.

## Viewing Training Progress

During the RL agent's training, logs are saved for visualization. You can monitor the training process using TensorBoard.

Run the following command to start the TensorBoard server:
```bash
uv run tensorboard --logdir ./ppo_steel_tensorboard/ --port 6006
```

Then, open your web browser and navigate to [http://localhost:6006](http://localhost:6006) to view the metrics.
