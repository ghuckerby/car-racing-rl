# Car Racing RL

Project for Designing Intelligent Agents Coursework

# Planning:

1. Background research on papers
2. Develop research questions
3. Implement
4. Evaluate using 20+ episodes
5. Clearly document usage of stable_baselines3 and gymnasium (what's mine and what's taken)

## Research Question Ideas:

1. Reward Shaping and Driving Style Analysis:

How do different reward function designs influence the driving policy, and can reward shaping produce distincts driving 'styles'?

- Custom wrapper subclasses that replace the default reward (Speed, efficiency, safety, time pressure)
- Trajectory analysis
- Quantitative metrics: average speed, steering variance, off-track %, lap completion rate, total reward
- Visualisations: overlay driving lines on the track, speed heatmaps, radar/spider charts comparing styles
- Statistical testing across 20+ evaluation episodes per agent
