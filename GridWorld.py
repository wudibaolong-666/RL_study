import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from typing import Optional
import time
from gym.envs.classic_control import rendering



class GridWorld():

    def __init__(self, world_length=5, seed=None):
        self.world_length = world_length
        self.seed(seed)
        self.viewer = None
        self.person = None
        self.person_transform = None

        self.x_pos = 0
        self.y_pos = 0
        self.state = np.array([self.x_pos, self.y_pos], dtype=np.int32)

        move = [
            [0,0],        # stop
            [0,1],        # right
            [1,0],       # up
            [0,-1],       # left
            [-1,0]         # down
        ]
        self.move = np.array(move, dtype=np.int32)

        grid_map = np.zeros((self.world_length,self.world_length))
        # set wall
        for i in range(self.world_length):
            for j in range(self.world_length):
                random_num = np.random.uniform(0,1)
                if random_num > 0.8:
                    grid_map[i, j] = 1
        # set goal place
        grid_map[self.world_length-1, self.world_length-1] = 0
        grid_map[0, 0] = 0

        self.grid_map = grid_map
        self.action_space = spaces.Discrete(5)
        self.observation_space =  spaces.MultiDiscrete([self.world_length, self.world_length])


    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    '''
     r = 1     get goal place
     r = 0     
     r = -0.01    get boundary or wall
        
    '''
    def step(self, action):
        # check action
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        move_action = self.move[action]
        position = self.state

        # move
        position_new = position + move_action
        if 0 <= position_new[0] < self.world_length:
            self.state[0] = position_new[0]
        if 0 <= position_new[1] < self.world_length:
            self.state[1] = position_new[1]

        # get reward and done
        done = 0
        if 0 <= position_new[0] < self.world_length and 0 <= position_new[1] < self.world_length:
            if np.array_equal(position_new , [self.world_length-1 ,self.world_length-1]):
                reward = 1
                done = 1
            elif self.grid_map[position_new[0], position_new[1]] == 0:
                reward = 0
            else:
                reward = -0.01
        else:
            reward = -1

        return self.state, reward, done

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = np.array([self.x_pos, self.y_pos], dtype=np.int32)
        return self.state

    def render(self):
        screen_width = 600
        screen_height = 600

        if self.viewer is None:
            # 初始化 Viewer
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 计算每个单元格的宽度和高度
            self.cell_width = screen_width / self.world_length
            self.cell_height = screen_height / self.world_length

            # 绘制方格
            for i in range(self.world_length):
                for j in range(self.world_length):
                    # 计算每个网格单元的左下角和右上角的坐标
                    l, r = j * self.cell_width, (j + 1) * self.cell_width  # 左右坐标
                    b, t = i * self.cell_height, (i + 1) * self.cell_height  # 上下坐标

                    # 使用 rendering.FilledPolygon 绘制矩形方格
                    cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

                    # 给每个方格设置颜色
                    if self.grid_map[i, j] == 0:
                        cell.set_color(1, 1, 1)  # 白色
                    else:
                        cell.set_color(0.5, 0.5, 0)  # 棕色

                    # 将每个方格添加到 viewer 中
                    self.viewer.add_geom(cell)

                    # 添加网格线
                    line_left = rendering.Line((l, b), (l, t))
                    line_right = rendering.Line((r, b), (r, t))
                    line_top = rendering.Line((l, t), (r, t))
                    line_bottom = rendering.Line((l, b), (r, b))

                    # 设置线条颜色为黑色
                    line_left.set_color(0, 0, 0)
                    line_right.set_color(0, 0, 0)
                    line_top.set_color(0, 0, 0)
                    line_bottom.set_color(0, 0, 0)

                    # 将网格线添加到 viewer 中
                    self.viewer.add_geom(line_left)
                    self.viewer.add_geom(line_right)
                    self.viewer.add_geom(line_top)
                    self.viewer.add_geom(line_bottom)

            # 创建人物
            person_radius = min(self.cell_width, self.cell_height) / 4  # 人物大小为单元格宽度/高度的1/4
            self.person = rendering.make_circle(person_radius)
            self.person.set_color(1, 0, 0)  # 红色
            self.person_transform = rendering.Transform()
            self.person.add_attr(self.person_transform)
            self.viewer.add_geom(self.person)

        # 每次 render 更新人物的位置
        person_i = self.state[0]
        person_j = self.state[1]

        # 更新人物的中心位置
        person_center_x = (person_j + 0.5) * self.cell_width
        person_center_y = (person_i + 0.5) * self.cell_height

        # 动态更新人物的平移属性
        self.person_transform.set_translation(person_center_x, person_center_y)

        return self.viewer.render(return_rgb_array=False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def set_state(self, state):
        self.state = state


if __name__ == "__main__":
    # 初始化 GridWorld 环境
    env = GridWorld(world_length=4)

    # 重置环境，获得初始状态
    state = env.reset()
    print(f"Initial state: {state}")

    for _ in range(10):  # 假设运行100步
        env.render()  # 渲染环境
        time.sleep(0.5)  # 控制帧速率，避免刷新过快

        # 假设用某种策略更新人物状态
        action = 2  # 随机选择一个动作
        env.step(action)  # 执行动作并更新状态

    env.close()





























