from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True


def calc_potential_field(goal, obstacles, resolution, rr, start):
    start_x, start_y = start
    goal_x, goal_y = goal
    obstacles_x, obstacles_y = obstacles
    minx = min(min(obstacles_x), start_x, goal_x) - AREA_WIDTH / 2.0
    miny = min(min(obstacles_y), start_y, goal_y) - AREA_WIDTH / 2.0
    maxx = max(max(obstacles_x), start_x, goal_x) + AREA_WIDTH / 2.0
    maxy = max(max(obstacles_y), start_y, goal_y) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / resolution))
    yw = int(round((maxy - miny) / resolution))

    # calc each potential
    potential_map = [[0.0 for _ in range(yw)] for _ in range(xw)]

    for ix in range(xw):
        x = ix * resolution + minx

        for iy in range(yw):
            y = iy * resolution + miny
            attractive_potential = calc_attractive_potential(x, y, goal)
            repulsive_potential = calc_repulsive_potential(x, y, obstacles, rr)
            total_potential = attractive_potential + repulsive_potential
            potential_map[ix][iy] = total_potential

    return potential_map, minx, miny


def calc_attractive_potential(x, y, goal):
    goal_x, goal_y = goal
    return 0.5 * KP * np.hypot(x - goal_x, y - goal_y)


def calc_repulsive_potential(x, y, obstacles, rr):
    # search nearest obstacle
    obstacles_x, obstacles_y = obstacles
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(obstacles_x):
        d = np.hypot(x - obstacles_x[i], y - obstacles_y[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - obstacles_x[minid], y - obstacles_y[minid])

    if dq <= rr:
        dq = max(dq, 0.1)
        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def actions():
    # dx, dy
    action_model = [[1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1]]

    return action_model


def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH:
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(start, goal, obstacles, resolution, rr):
    # calc potential field
    start_x, start_y = start
    goal_x, goal_y = goal
    pmap, minx, miny = calc_potential_field(goal, obstacles, resolution, rr, start)

    # search path
    d = np.hypot(start_x - goal_x, start_y - goal_y)
    ix = round((start_x - minx) / resolution)
    iy = round((start_y - miny) / resolution)
    gix = round((goal_x - minx) / resolution)
    giy = round((goal_y - miny) / resolution)

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [start_x], [start_y]
    action_space = actions()
    previous_ids = deque()

    while d >= resolution:
        minp = float("inf")
        minix, miniy = -1, -1
        for i, _ in enumerate(action_space):
            inx = int(ix + action_space[i][0])
            iny = int(iy + action_space[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * resolution + minx
        yp = iy * resolution + miny
        d = np.hypot(goal_x - xp, goal_y - yp)
        rx.append(xp)
        ry.append(yp)

        if oscillations_detection(previous_ids, ix, iy):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        if show_animation:
            plt.plot(ix, iy, ".r")
            plt.pause(0.01)

    print("Goal!!")

    return rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.grid(False)
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")
    start = 0.0, 10.0
    goal = 30.0, 30.0
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]
    obstacles_x = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
    obstacles_y = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]
    obstacles = obstacles_x, obstacles_y
    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    _, _ = potential_field_planning(
        start, goal, obstacles, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    main()
