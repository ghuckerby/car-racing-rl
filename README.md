# Car Racing RL

Project for Designing Intelligent Agents Coursework

# Planning:

1. Background research on papers
2. Develop research questions
3. Implement
4. Evaluate using 20+ episodes
5. Clearly document usage of stable_baselines3 and gymnasium (what's mine and what's taken)

## Research Question Ideas:

1. Comparison of Action Spaces (Continuous vs Discrete) on driving style.

- Train an agent with the same algorithm using different action spaces
- Compare trajectories, average rewards, lap completion, and qualitative findings.

2. Generalisation and Transfer Learning

- How does an agent train on one/standard tracks generalise to others?
- Train on standard tracks then evaluate on modified environment versions.
- (Investigate track randomisation and custom creation)

3. Perception and Frame Stacking

- How does the amount of visual information impact the agent's ability to handle high-speed cornering
- Train agent with different stacking depths and measure performance on sharp turns vs straight sections.

4. Reward Shaping Effect

- Default reward in Car Racing is tiles visited
- How do different reward components influence the learned driving policy?
- Create custom environment wrappers that modify the reward function
- Train seperate agents on these 'styles' and use visualisations to compare driving lines and lap times
