import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation
from function import  list_to_tuple
import random

class Map:
    class Nodes:
        ''' Class for representing the nodes used by the ACO algorithm '''
        def __init__(self, row, col, in_map, spec):
            self.node_pos = (row, col)
            self.edges = self.compute_edges(in_map)
            self.spec = spec

        def compute_edges(self, map_arr):
            ''' class that returns the edges connected to each node '''
            imax = map_arr.shape[0]  # map_arr的行数
            jmax = map_arr.shape[1]  # map_arr的列数
            edges = []
            if map_arr[self.node_pos[0]][self.node_pos[1]] == 1:
                for dj in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        newi = self.node_pos[0] + di
                        newj = self.node_pos[1] + dj
                        if (dj == 0 and di == 0):
                            continue
                        if (newj >= 0 and newj < jmax and newi >= 0 and newi < imax):
                            if map_arr[newi][newj] == 1:  # 无障碍物 且 一步可达
                                edges.append((newi, newj))
            return edges

    def __init__(self, map_name):
        self.in_map = self._read_map(map_name)
        self.occupancy_map = self._map_2_occupancy_map()
        self.map_avaliables= self.add_avaliable_nodes()
        self.start_position = (0,0)
        self.map_avaliables.remove(self.start_position)
        self.end_position =(5,5)
        self.barrier = self.add_obs_nodes()
        self.nodes_array = self._create_nodes()
        self.height = self.occupancy_map.shape[0]
        self.width = self.occupancy_map.shape[1]


    def obs_map(self):
        x = self.x_range
        y = self.y_range
        obs = set()
        obs_nodes = self.add_obs_nodes()
        for i in obs_nodes:
            i = list(i)
            i[0] = i[0] + 1
            i[1] = i[1] + 1
            obs.add((i[0] ,i[1] ))
        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))
        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))
        return obs

    def init_map(self):
        self.start_position = tuple(random.sample(self.map_avaliables, 1)[0])
        self.map_avaliables.remove(self.start_position)
        self.end_position = tuple(random.sample(self.map_avaliables, 1)[0])
    def add_avaliable_nodes(self):
        avaliable_nodes = []
        nodes = []
        points = np.where(self.in_map == 'E')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            avaliable_nodes.append(nodes)
            nodes = []
        return list_to_tuple(avaliable_nodes)

    def _create_nodes(self):
        return [[self.Nodes(i, j, self.occupancy_map, self.in_map[i][j]) for j in
                 range(self.in_map.shape[1])] for i in range(self.in_map.shape[0])]

    def _read_map(self, map_name):
        in_map = np.loadtxt('./maps/' + map_name, dtype=str )
        return in_map

    def add_initial_node(self):
        initial_node = []
        nodes = []
        points = np.where(self.in_map == 'S')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            initial_node.append(nodes)
            nodes = []
        return initial_node

    def add_final_node(self):
        final_node = []
        nodes = []
        points = np.where(self.in_map == 'F')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            final_node.append(nodes)
            nodes = []
        return final_node

    def add_obs_nodes(self):
        obs_nodes = []
        nodes = []
        points = np.where(self.in_map == 'O')
        for i in range(len(points[1])):
            for point in points:
                node = int(point[i])
                nodes.append(node)
            obs_nodes.append(nodes)
            nodes = []
        return list_to_tuple(obs_nodes)

    def _map_2_occupancy_map(self):
        map_arr = np.copy(self.in_map)
        map_arr[map_arr == 'O'] = 0
        map_arr[map_arr == 'E'] = 1
        return map_arr.astype(np.int)

    def represent_map(self):
        plt.plot(self.start_position[1], self.start_position[0], 'bo', markersize=10)
        plt.plot(self.end_position[1], self.end_position[0], 'r*', markersize=10)
        plt.imshow(self.occupancy_map, cmap='gray', interpolation = 'nearest')
        plt.grid(ls="--")
        plt.show()

    def represent_path(self, path):
        shape = list(copy.deepcopy(self.occupancy_map.shape))
        width = shape[0]
        height = shape[1]
        x = []
        y = []
        for p in path:
            x.append(p[1])
            y.append(p[0])
        plt.figure(figsize=(width,height))
        plt.plot(x, y)
        plt.grid(ls="--")
        plt.plot(self.start_position[1], self.start_position[0], 'bo', markersize=10)
        plt.plot(self.end_position[1], self.end_position[0], 'r*', markersize=10)
        plt.imshow(self.occupancy_map, cmap='gray', interpolation = 'nearest')
        plt.show()
        plt.close()
        # 开始制作动画
        # fig = plt.figure()
        # point_ani, = plt.plot(x[0], y[0], "go")
        # plt.imshow(self.occupancy_map, cmap='gray', interpolation='nearest')
        # plt.grid(ls="--")
        # text_pt = plt.text(4, 0.8, '', fontsize=12, color='green')
        #
        # ani = animation.FuncAnimation(fig, update_points, np.arange(0, (len(x))), interval=1000, blit=True)
        # # ani.save('test.gif', writer='imagemagick', fps=1)
        # plt.show()
        # plt.close()


