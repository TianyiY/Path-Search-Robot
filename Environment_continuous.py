'''
Constructing the maze environment and an automatic path searching robot
'''

import numpy as np
import pyglet      # for painting

barriers_collection = {'SW_x': [250, 450], 'SW_y':[0, 300],    # make sure SW_y==SE_y;  NE_y==NW_y
                       'NW_x': [250, 450], 'NW_y':[500, 800],    #           SW_x==NW_x;  SE_x==NE_x
                       'NE_x': [300, 500], 'NE_y':[500, 800],    # rectangle
                       'SE_x': [300, 500], 'SE_y':[0, 300]}    # four corners of barriers
barriers = []
screen_len=800

class Robot(object):

    visible = None
    refresh_frequency = 1.    # time interval for refreshing the movement
    target = {'x_coor': 750., 'y_coor': 750.}      # target range
    state_N = 2   # (robot_x, robot_y)
    action_N = 1 # (orientation)
    action_limit = [0, 2.*np.pi]    # range of orientation change

    def __init__(self):
        self.robot_info = np.zeros(1, dtype=[('x', np.float16), ('y', np.float16), ('l', np.float16)])
        self.robot_info['x'] = 50    # robot center location x coordinate
        self.robot_info['y'] = 50    # robot center location y coordinate
        self.robot_info['l'] = 10    # robot length
        self.within_target=0

    def step(self, action):

        done = False

        # initialize rewards
        reward = -np.sqrt((0.001 * self.robot_info['x'] - 0.001 * self.target['x_coor']) ** 2 +
                          (0.001 * self.robot_info['y'] - 0.001 * self.target['y_coor']) ** 2)

        action = np.clip(action, *self.action_limit)
        dx=np.cos(action) * self.refresh_frequency    # calculate step length in x
        dy=np.sin(action) * self.refresh_frequency    # calculate step length in y

        if (self.robot_info['l']/2)<=(self.robot_info['x']+dx) and (self.robot_info['x']+dx)<=(screen_len-(self.robot_info['l']/2)) \
                and (self.robot_info['l'] / 2)<=(self.robot_info['y']+dy) and (self.robot_info['y']+dy) <= (screen_len - (self.robot_info['l'] / 2)):
            if not (((barriers_collection['SW_x'][0] - self.robot_info['l'] / 2) < (self.robot_info['x'] + dx) and
                            (self.robot_info['x'] + dx) < (
                                    barriers_collection['SE_x'][0] + self.robot_info['l'] / 2) and
                            (barriers_collection['SW_y'][0] - self.robot_info['l'] / 2) < (
                                    self.robot_info['y'] + dy) and
                            (self.robot_info['y'] + dy) < (
                                    barriers_collection['NW_y'][0] + self.robot_info['l'] / 2)) or \
                    ((barriers_collection['SW_x'][1] - self.robot_info['l'] / 2) < (self.robot_info['x'] + dx) and
                             (self.robot_info['x'] + dx) < (
                                     barriers_collection['SE_x'][1] + self.robot_info['l'] / 2) and
                             (barriers_collection['SW_y'][1] - self.robot_info['l'] / 2) < (
                                     self.robot_info['y'] + dy) and
                             (self.robot_info['y'] + dy) < (
                                     barriers_collection['NW_y'][1] + self.robot_info['l'] / 2))):
                self.robot_info['x'] += dx
                self.robot_info['y'] += dy  # update robot position info

                # done and reward
                if (self.target['x_coor'] + self.robot_info['l'] / 2) < self.robot_info['x'] and \
                                self.robot_info['x'] < (screen_len - self.robot_info['l'] / 2) and \
                                (self.target['y_coor'] + self.robot_info['l'] / 2) < self.robot_info['y'] and \
                                self.robot_info['y'] < (screen_len - self.robot_info['l'] / 2):
                    reward += 100
                    self.within_target += 1
                    if self.within_target > 10:
                        done = True

            else:
                reward -= 1
        else:
            reward-=1

        # state
        state = np.array([self.robot_info['x'], self.robot_info['y']]).flatten()
        return state, reward, done

    def reset(self):
        self.robot_info['x'] = 50  # robot center location x coordinate
        self.robot_info['y'] = 50  # robot center location y coordinate
        self.robot_info['l'] = 10  # robot length
        self.within_target = 0

        # state
        state = np.array([self.robot_info['x'], self.robot_info['y']]).flatten()
        return state

    def render(self):
        if self.visible is None:
            self.visible = Visualization(self.robot_info, self.target)
        self.visible.render()

    def random_choose_action(self):
        action=2.*np.random.rand(1)*np.pi    # sample action
        return action


class Visualization(pyglet.window.Window):

    def __init__(self, robot_info, target):
        super(Visualization, self).__init__(width=screen_len, height=screen_len, resizable=False, caption='robot', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.robot_info = robot_info
        self.batch = pyglet.graphics.Batch()    # display as whole batch


        # target batch
        self.target = self.batch.add(4, pyglet.gl.GL_QUADS, None,
            # 4 corners
            ('v2f', [target['x_coor'], target['y_coor'],                # location
                     target['x_coor'], screen_len,
                     screen_len,  screen_len,
                     screen_len,  target['y_coor']]),
            ('c3B', (60, 249, 60) * 4))    # color   R/G/B

        # robot batch
        self.robot = self.batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.robot_info['x'] - self.robot_info['l'] / 2, self.robot_info['y'] - self.robot_info['l'] / 2,                # location
                     self.robot_info['x'] - self.robot_info['l'] / 2, self.robot_info['y'] + self.robot_info['l'] / 2,
                     self.robot_info['x'] + self.robot_info['l'] / 2, self.robot_info['y'] + self.robot_info['l'] / 2,
                     self.robot_info['x'] + self.robot_info['l'] / 2, self.robot_info['y'] - self.robot_info['l'] / 2]),
            ('c3B', (249, 60, 60) * 4,))    # color

        # barrier batches
        for i in range(len(barriers_collection['SW_x'])):
            self.barrier_iter = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                           ('v2f', [barriers_collection['SW_x'][i], barriers_collection['SW_y'][i],  # location
                                                    barriers_collection['NW_x'][i], barriers_collection['NW_y'][i],
                                                    barriers_collection['NE_x'][i], barriers_collection['NE_y'][i],
                                                    barriers_collection['SE_x'][i], barriers_collection['SE_y'][i]]),
                                           ('c3B', (0, 0, 0) * 4,))  # color
            barriers.append(self.barrier_iter)

    def render(self):
        self.update_robot()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def update_robot(self):
        xx = self.robot_info['x']
        yy = self.robot_info['y']
        SW_xx=xx-self.robot_info['l']/2
        SW_yy=yy-self.robot_info['l']/2
        NW_xx=xx-self.robot_info['l']/2
        NW_yy=yy+self.robot_info['l']/2
        NE_xx=xx+self.robot_info['l']/2
        NE_yy=yy+self.robot_info['l']/2
        SE_xx=xx+self.robot_info['l']/2
        SE_yy=yy-self.robot_info['l']/2

        self.robot.vertices = [SW_xx, SW_yy, NW_xx, NW_yy, NE_xx, NE_yy, SE_xx, SE_yy]

if __name__ == '__main__':
    robot = Robot()
    while True:
        robot.render()
        action=robot.random_choose_action()
        print('orientation: ', action)
        state, reward, _=robot.step(action)
        print('coordinate: ', state)
        print('reward: ', reward)