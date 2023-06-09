import matplotlib.pyplot as plt
import random

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def plot_picture(display,route,n,map):
    if display > 0:
        for path in route:
            x = []
            y = []
            for p in path:
                x.append(p[1])
                y.append(p[0])
            plt.plot(x, y, marker='o', color=randomcolor(), markersize=4)

        for i in range(n):
            plt.plot(map.initial_node[i][1], map.initial_node[i][0], 'bo', markersize=8)
            plt.plot(map.final_node[i][1], map.final_node[i][0], 'r*', markersize=8)

    plt.imshow(map.occupancy_map, cmap='gray', interpolation='nearest')
    plt.show()
    plt.close()


def motion_move(route):
    route_sort = sorted(route, key=lambda i: len(i),reverse=True)
    max = len(route_sort[1])
    for i in range(max):
        for j in range(len(route_sort)):
          plt.plot(route[i])
        plt.pause(1)
