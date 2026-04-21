# Car Racing RL

Project for Designing Intelligent Agents Coursework

## Proposal:

My proposed project will investigate how reward function design affects the behaviour and performance of reinforcement learning driving agents in the OpenAI Gym/Gymnasium CarRacing-v3 environment.

Using PPO from Stable-Baselines3 as the baseline agent, the project will implement and compare several reward function wrappers that focus on different objectives in the environment, such as progress, speed, safety, control smoothness, and time. These reward designs will be evaluated through controlled experiments using fixed training budgets and multiple seeds for reliable comparisons. In addition to standard performance metrics, the project will collect telemetry and trajectory data to analyse behavioural characteristics and navigation of the racetrack environment. The aim is to assess how different reward designs produce different driving styles and assess their performance trade-offs.

As an extension, a search-based meta-learning approach will be implemented to automatically optimise reward weights for each wrapper, allowing comparison between the individual reward function wrappers and a discovered reward configuration. This provides a further investigation into reward design in the reinforcement learning system.
