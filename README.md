# Path-Search-Robot
Training a robot which can find its path to the target by deep Q learning using self built simulator

1. Environment
- Built a virtual robot, a virtual target and virtual barriers. 
- Visualized by pyglet 
- Two features in state (maybe more rational features should be added later)
- Continuous action (based on orientation of virtual robot)
- Discrete action (right, up, left, down)
- Reward according to the normalized distance between target and robot, add 100 reward once achieving target, minus 1 reward if colliding with barriers 
