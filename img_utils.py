import math
import time

import matplotlib.pyplot as plt
import numpy as np


def show_point_cloud(points_x, points_y, points_z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Point cloud at {} sec. ".format(str(int(time.time()))))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.plot(0, 0, 0, 'o', color="black", markersize=2.0)

    ax.plot((points_x), (points_y), (points_z), 'o', color="black", markersize=0.5)

    ax.view_init(elev=10.0, azim=None)

    show_or_close(True)


def show_or_close(show):
    if show:
        plt.show(block=False)
    else:
        plt.close()


def get_degrees_between_normals(normals):

    size = normals.shape[0]
    degrees = []
    for i in range(size - 1):
        for j in range(i + 1, size):
            x = normals[i] / np.linalg.norm(normals[i])
            y = normals[j] / np.linalg.norm(normals[j])
            degrees.append(math.acos(np.dot(x, y)) * 180 / math.pi)
    return degrees


def show_and_save_normal_clusters_3d(normals, clustered_normals, normal_indices, show, save, out_dir, img_name):

    if not show and not save:
        return

    cluster_color_names = ["red", "green", "blue"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    degrees_list = get_degrees_between_normals(clustered_normals)
    title = "Normals clustering: {}".format(img_name)
    if len(degrees_list) > 0:
        title = title + "\n degrees: {}".format(",\n".join([str(s) for s in degrees_list]))
    plt.title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    ax.plot(0, 0, 0, 'o', color="black", markersize=2.0)

    rel_normals = normals[normal_indices == 3]
    ax.plot((rel_normals[::10, 0]), (rel_normals[::10, 2]), (rel_normals[::10, 1]), '.', color="yellow", markersize=0.5)
    ax.plot((-rel_normals[::10, 0]), (-rel_normals[::10, 2]), (-rel_normals[::10, 1]), '.', color="yellow", markersize=0.5)

    for i in range(len(clustered_normals)):
        rel_normals = normals[normal_indices == i]
        ax.plot((rel_normals[::10, 0]), (rel_normals[::10, 2]), (rel_normals[::10, 1]), '.', color=cluster_color_names[i], markersize=0.5)
        ax.plot((-rel_normals[::10, 0]), (-rel_normals[::10, 2]), (-rel_normals[::10, 1]), '.', color=cluster_color_names[i], markersize=0.5)

    if len(clustered_normals.shape) == 1:
        clustered_normals = np.expand_dims(clustered_normals, axis=0)

    for i in range(len(clustered_normals)):
        ax.plot((clustered_normals[i, 0]), (clustered_normals[i, 2]), (clustered_normals[i, 1]), 'o', color="black", markersize=5.0)

    ax.view_init(elev=10.0, azim=None)

    x_lim = [-1, 1]
    y_lim = [-1, 1]
    z_lim = [-1, 1]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    if show:
        for i in range(clustered_normals.shape[0] - 1):
            for j in range(i + 1, clustered_normals.shape[0]):
                angle = np.arccos(np.dot(clustered_normals[i], clustered_normals[j]))
                angle_degrees = 180 / math.pi * angle
                print("angle between normal {} and {}: {} degrees".format(i, j, angle_degrees))

    # NOTE: first save, then show
    if save:
        out_path = '{}/{}_point_cloud.jpg'.format(out_dir, img_name[:-4])
        plt.savefig(out_path)

    show_or_close(show)

