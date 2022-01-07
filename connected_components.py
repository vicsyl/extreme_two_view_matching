import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from img_utils import show_or_close
from utils import Timer, merge_keys_for_same_value


def get_and_show_components(cluster_indices, valid_component_dict, title=None, normals=None, show=True, save=False, path=None, file_name=None):

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
    ]

    color_names = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "maroon",
        "dark green",
        "navy"
    ]

    cluster_colors = np.zeros((cluster_indices.shape[0], cluster_indices.shape[1], 3), dtype=np.int32)
    for i, c_index in enumerate(valid_component_dict.keys()):
        cluster_colors[np.where(cluster_indices == c_index)] = colors[i % 9]

    plt.figure(figsize=(9, 9))

    if title is not None:
        plt.title(title)
    elif normals is not None:
        title = "{} - connected components: \n".format(file_name)
        new_component_dict = {}
        for i, c_index in enumerate(valid_component_dict.keys()):
            new_component_dict[i] = valid_component_dict[c_index]
        merged_dict = merge_keys_for_same_value(new_component_dict)
        for merged_values in merged_dict:
            cur_colors_names = ", ".join([color_names[val % 9] for val in merged_values])
            title = "{}[{}]={}={},\n".format(title, cur_colors_names, normals[merged_dict[merged_values]], merged_dict[merged_values])
        plt.title(title)

    plt.imshow(cluster_colors)
    if save:
        plt.savefig(path)

    show_or_close(show)
    return cluster_colors


def circle_like_ones(size):
    ret = np.ones((size, size), np.uint8)
    r_check = (size / 2 - 0.4) ** 2
    for i in range(size):
        for j in range(size):
            r = (size / 2 - (i + 0.5)) ** 2 + (size / 2 - (j + 0.5)) ** 2
            if r > r_check:
                ret[i, j] = 0
    return ret


def flood_fill(input_img):

    flood_filled = input_img.copy()
    flood_filled[0, :] = 0
    flood_filled[flood_filled.shape[0] - 1, :] = 0
    flood_filled[:, flood_filled.shape[1] - 1] = 0
    flood_filled[:, 0] = 0

    mask = np.zeros((flood_filled.shape[0] + 2, flood_filled.shape[1] + 2), np.uint8)
    cv.floodFill(flood_filled, mask, (0, 0), 2)
    flood_filled = np.where(flood_filled == 2, 0, 1).astype(dtype=np.uint8)
    flood_filled = flood_filled | input_img
    return flood_filled


def get_connected_components(normal_indices, valid_indices, show=False,
                             fraction_threshold=0.03, closing_size=None, flood_filling=False, connectivity=4):

    Timer.start_check_point("get_connected_components")

    component_size_threshold = normal_indices.shape[0] * normal_indices.shape[1] * fraction_threshold

    out = np.zeros((normal_indices.shape[0], normal_indices.shape[1]), dtype=np.int32)
    out_valid_indices_dict = {}
    out_valid_indices_counter = 0

    for v_i in valid_indices:
        input = np.where(normal_indices == v_i, 1, 0).astype(dtype=np.uint8)

        if closing_size is not None:
            kernel = circle_like_ones(size=closing_size) # np.ones((closing_size, closing_size) np.uint8)
            input = cv.morphologyEx(input, cv.MORPH_CLOSE, kernel)

        if flood_filling:
            input = flood_fill(input)

        ret, labels = cv.connectedComponents(input, connectivity=connectivity)

        unique, counts = np.unique(labels, return_counts=True)
        valid_labels = np.where(counts > component_size_threshold)[0]
        # Docs: RETURNS: The sorted unique values. - see https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        if valid_labels[0] == 0:
            valid_labels = valid_labels[1:]
        if len(valid_labels) != 0:
            max_valid_labels = np.max(valid_labels)
            valid_labels = valid_labels + out_valid_indices_counter
            labels = labels + out_valid_indices_counter

            for v_i_i in valid_labels:
                out = np.where(labels == v_i_i, labels, out)

            out_valid_indices_dict.update({v_i_i: v_i for v_i_i in valid_labels})
            out_valid_indices_counter = out_valid_indices_counter + max_valid_labels

        if show:
            get_and_show_components(out, out_valid_indices_dict, "out after normal index={}".format(v_i))

    Timer.end_check_point("get_connected_components")

    return out, out_valid_indices_dict
