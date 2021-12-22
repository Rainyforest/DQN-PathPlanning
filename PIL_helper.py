from matplotlib import pyplot as plt


def draw_mat(mat):
    plt.imshow(mat, cmap="binary")


def draw_point(x, y, size=10, color='k', label=''):
    plt.plot(x, y, marker='o', markersize=size, color=color, label=label, linestyle="None")
