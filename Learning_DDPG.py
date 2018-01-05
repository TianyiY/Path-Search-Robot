from Environment_continuous import Robot
from Brain_DDPG_AC import brain

EPISODES = 500
STEPS = 2000
ON_TRAIN = True

# set env
robot = Robot()
state_N = robot.state_N
action_N = robot.action_N
action_limit = robot.action_limit

# set RL method (continuous)
RL_brain = brain(action_N, state_N, action_limit)

def train():
    # start training
    for i in range(EPISODES):
        state = robot.reset()
        total_rewards = 0.

        for j in range(STEPS):
            robot.render()
            action = RL_brain.choose_action(state)
            state_, reward, done = robot.step(action)
            RL_brain.store(state, action, reward, state_)
            total_rewards += reward
            #print('action: ', action, 'state: ', state, 'reward: ', reward)

            if RL_brain.is_memory_full:
                # start to learn once has fulfilled the memory
                RL_brain.learning()
            state = state_

            if done or j == STEPS-1:
                print('Episodes: %i | %s | total reward: %.1f | step: %i' % (i, '---' if not done else 'done', total_rewards, j))
                break
    RL_brain.save()

def eval():
    RL_brain.restore()
    robot.render()
    robot.visible.set_vsync(True)
    while True:
        state = robot.reset()
        for _ in range(2000):
            robot.render()
            action = RL_brain.choose_action(state)
            state, reward, done = robot.step(action)
            if done:
                break

if ON_TRAIN:
    train()
else:
    eval()