# Reinforcement Learning Midterm Project - Pablo Salar Carrera
## Gambler's Problem
This repository contains a solution to the **Gambler's Problem** from *Reinforcement Learning: An Introduction* (Barto & Sutton). The goal of the project is to solve the problem using **Dynamic Programming** techniques, specifically **Value Iteration** and **Policy Iteration**.

The project includes two main tasks:
1. **Task 1**: Solve the Gambler’s problem for \( p_h = 0.25 \) and \( p_h = 0.55 \) using **Value Iteration** and **Policy Iteration**.
2. **Task 2**: Modify the problem rules and solve the modified problem, analyzing how changes in betting strategy and reward structure affect the policy.

## Problem Overview

The **Gambler's Problem** is an episodic, finite Markov Decision Process (MDP), where the gambler makes bets on a sequence of coin flips. The gambler's capital is represented by the state, and the action is the amount of money staked on each flip. The goal is to maximize the probability of reaching a capital of $100 before running out of money.

- The **states** represent the gambler's capital, \( s ∈ \{1, 2, 3, ..., 99\} \), with terminal states at \( s = 0 \) (gambling loses) and \( s = 100 \) (gambling wins).
- The **actions** are stakes, \( a ∈ \{0, 1, 2, ..., \min(s, 100 - s)\} \), meaning the gambler can stake between 0 and the minimum of their current capital \( s \) or the amount needed to reach $100.
- The **reward** is 0 except when the gambler reaches $100, at which point the reward is +1.
- The problem is solved using **Value Iteration** and **Policy Iteration**.



In **Task 2**, the rules are modified:
- The gambler can now stake up to the full amount of their capital.
- A special rule is introduced where, if the coin lands heads, there is a \(1/8\) chance the gambler's winnings are doubled (North head).
- The reward for winning is the final capital minus 99 (e.g., if the gambler reaches $120, the reward is \(120 - 99 = 21\)).

## Project Structure

## Approach

### Task 1: Value Iteration and Policy Iteration

- **Value Iteration**: Iteratively updates the state values using the Bellman equation:
  \[
  V(s) = \max_a \left( p_h · V(s + a) + (1 - p_h) · V(s - a) \right)
  \]
  Where:
- \( V(s) \) is the value of the state \( s \),
- \( a \) is the action taken,
- \( p_h \) is the probability of heads (the coin flip probability),
- \( V(s + a) \) and \( V(s - a) \) are the values of the next state after winning or losing the bet, respectively.
- **Policy Iteration**: Alternates between policy evaluation and policy improvement, refining the policy and value function until convergence.

### Task 2: Modified Gambling Rules

- The gambler can stake the full amount of their current capital.
- The reward function is modified to reflect the new capital.
- A special "North head" outcome gives a \(1/8\) chance of doubling the winnings when heads is flipped.
- The state transition and reward function are updated to account for these changes.

### Key Parameters

- \( p_h = 0.25 \) and \( p_h = 0.55 \) are the probabilities of the coin landing heads.
- \( \gamma = 1.0, 0.99, 0.90 \) are the discount factors used to explore the effect of future rewards.

## Results and Analysis

- The **optimal policy** and **state values** for both \( p_h = 0.25 \) and \( p_h = 0.55 \) are plotted.
- **Policy Iteration** and **Value Iteration** are compared to determine the best strategy for the gambler under different conditions.
- **Discount Factor**: The effect of changing the discount factor (\( \gamma \)) on the optimal policy is analyzed.

**Figures** (saved in the `results/` folder):
- Policy and value plots for \( p_h = 0.25 \) and \( p_h = 0.55 \).
- Comparison of Value Iteration and Policy Iteration results.

## Folder Structure

The project is organized as follows:
- `/notebooks/`: Contains the Jupyter Notebook used for analysis.
  - `DS669MidtermProject_PabloSalar.ipynb` - Jupyter notebook with code, visualizations, and explanations.

- `/src/`: Contains the Python script used in the project.
  - `DS669MidtermProject_PabloSalar.py` - Standalone Python script for running the analysis.


