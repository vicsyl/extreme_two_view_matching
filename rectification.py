import math

import cv2 as cv
import kornia.geometry as KG
import matplotlib.pyplot as plt
import numpy as np
import torch

from img_utils import show_or_close
from resize import resample_nearest_numpy
from utils import get_rotation_matrix, get_rotation_matrix_safe


def get_rectification_rotation(normal, rotation_factor=1.0):

    # now the normals will be "from" me, "inside" the surfaces
    normal = -normal

    z = np.array([0.0, 0.0, 1.0])

    # this handles the case when there is only one dominating plane

    assert normal[2] > 0
    rotation_vector = np.cross(normal, z)
    rotation_vector_norm = abs_sin_theta = np.linalg.norm(rotation_vector)
    unit_rotation_vector = rotation_vector / rotation_vector_norm
    theta = math.asin(abs_sin_theta) * rotation_factor
    theta = min(theta, math.pi * 4.0/9.0)

    R = get_rotation_matrix(unit_rotation_vector, theta)
    det = np.linalg.det(R)
    assert math.fabs(det - 1.0) < 0.0001
    return R


def add_third_row(column_vecs):
    return np.vstack((column_vecs, np.ones(column_vecs.shape[1])))


def get_valid_mask(x_array, y_array, valid_box):
    mask = ((x_array >= valid_box[:,0].min()) &
            (x_array <= valid_box[:,0].max()) &
            (y_array >= valid_box[:,1].min()) &
            (y_array <= valid_box[:,1].max()))
    return mask


def get_valid_box(img, clip_angle, R, K):
    h, w = img.shape[:2]
    h_2, w_2 = h / 2.0, w / 2.0
    if clip_angle is None:
        clip_angle = 90
    angles_xyz = KG.rotation_matrix_to_angle_axis(torch.from_numpy(R)[None]).detach().cpu().numpy()[0]
    angles_xyz = np.rad2deg(angles_xyz)
    FOV_Y_2 = np.rad2deg(math.atan(h_2 / K[1][1]))
    FOV_X_2 = np.rad2deg(math.atan(w_2 / K[0][0]))

    dy = h_2 * (math.sin(np.deg2rad(clip_angle)) / math.sin(np.deg2rad(FOV_Y_2)))
    dx = w_2 * (math.sin(np.deg2rad(clip_angle)) / math.sin(np.deg2rad(FOV_X_2)))

    center_x = w_2 * (1.0 + (math.sin(np.deg2rad(angles_xyz[1])) / math.tan(np.deg2rad(FOV_X_2))))
    center_y = h_2 * (1.0 + (math.sin(np.deg2rad(angles_xyz[0])) / math.tan(np.deg2rad(FOV_Y_2))))

    valid_box = np.float32([[center_x - dx, center_y - dy],
                            [center_x - dx, center_y + dy],
                            [center_x + dx, center_y + dy],
                            [center_x + dx, center_y - dy]])
    return valid_box


def get_perspective_transform(img, R, K, K_inv, component_indices, index, clip_angle=None, scale=1.0):

    if clip_angle is not None:
        valid_box = get_valid_box(img, clip_angle, R, K)
        print(f'valid_box = {valid_box}')

    unscaled = True
    while unscaled:

        coords = np.where(component_indices == index)
        coords = np.array([coords[1], coords[0]])

        if clip_angle is not None:
            mask = get_valid_mask(coords[0], coords[1], valid_box)
            coords = coords[:, mask]

        coords = add_third_row(coords)

        P = K @ R @ K_inv
        if scale != 1.0:
            unscaled = False
            P[:2, :] *= scale

        new_coords = P @ coords
        new_coords = new_coords / new_coords[2, :]

        min_row = min(new_coords[1])
        max_row = max(new_coords[1])
        min_col = min(new_coords[0])
        max_col = max(new_coords[0])

        dst = np.float32([[min_col, min_row], [min_col, max_row - 1], [max_col - 1, max_row - 1], [max_col - 1, min_row]])
        dst = np.transpose(dst)
        dst = add_third_row(dst)

        if unscaled:
            new_bb_size = (max_row - min_row) * (max_col - min_col)
            scale = np.sqrt((coords.shape[1] * 2.0) / new_bb_size)
            if scale == 1.0:
                unscaled = False
                break

    translate_vec_new = (-np.min(dst[0]), -np.min(dst[1]))
    translate_matrix_new = np.array([
        [1, 0, translate_vec_new[0]],
        [0, 1, translate_vec_new[1]],
        [0, 0, 1],
    ])

    dst = translate_matrix_new @ dst
    P = translate_matrix_new @ P
    bounding_box_new = (math.ceil(np.max(dst[0])), math.ceil(np.max(dst[1])))

    return P, bounding_box_new


def get_rectified_keypoints(normals,
                            components_indices,
                            valid_components_dict,
                            img,
                            K,
                            descriptor,
                            img_name,
                            fixed_rotation_vector=None,
                            clip_angle=None,
                            show=False,
                            save=False,
                            out_prefix=None,
                            rotation_factor=1.0,
                            all_unrectified=False):

    K_inv = np.linalg.inv(K)

    all_descs = None
    all_kps = []

    for component_index in valid_components_dict:

        normal_index = valid_components_dict[component_index]
        normal = normals[normal_index]

        if fixed_rotation_vector is None:
            R = get_rectification_rotation(normal, rotation_factor)
        else:
            R = get_rotation_matrix_safe(fixed_rotation_vector * rotation_factor)

        T, bounding_box = get_perspective_transform(img, R, K, K_inv, components_indices, component_index, clip_angle)

        # NOTE this is to prevent out of memory errors, but actually never happens
        if bounding_box[0] * bounding_box[1] > 10**8:
            print("warping to an img that is too big, skipping")
            continue

        T_inv = np.linalg.inv(T)

        rectified = cv.warpPerspective(img, T, bounding_box)

        kps, descs = descriptor.detectAndCompute(rectified, None)

        kps_raw = np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)

        new_kps = cv.perspectiveTransform(kps_raw, T_inv)

        if new_kps is not None:
            kps_int_coords = np.int32(new_kps).reshape(-1, 2)

            h = img.shape[0]
            w = img.shape[1]
            first = kps_int_coords[:, 0]
            first = np.where(0 <= first, first, 0)
            first = np.where(first < w, first, 0)
            seconds = kps_int_coords[:, 1]
            seconds = np.where(0 <= seconds, seconds, 0)
            seconds = np.where(seconds < h, seconds, 0)
            kps_int_coords[:, 0] = first
            kps_int_coords[:, 1] = seconds

            cluster_mask_bool = np.array([components_indices[kps_int_coord[1], [kps_int_coord[0]]] == component_index for kps_int_coord in kps_int_coords])
            cluster_mask_bool = cluster_mask_bool.reshape(-1)

            descs = descs[cluster_mask_bool]
            new_kps = new_kps[cluster_mask_bool]

            kps = [kp for i, kp in enumerate(kps) if cluster_mask_bool[i]]

            cv.drawKeypoints(rectified, kps, rectified, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for kpi, kp in enumerate(kps):
                kp.pt = tuple(new_kps[kpi, 0].tolist())

            all_kps.extend(kps)

            if all_descs is None:
                all_descs = descs
            else:
                all_descs = np.vstack((all_descs, descs))

        if show or save:
            plt.figure(figsize=(9, 9))
            plt.title("{} - component: {},\n normal: {}".format(img_name, component_index, normals[normal_index]))
            plt.imshow(rectified)
            if save:
                plt.savefig("{}_rectified_component_{}".format(out_prefix, component_index))
            show_or_close(show)

    kps, descs = descriptor.detectAndCompute(img, None)

    if not all_unrectified:

        kps_floats = np.float32([kp.pt for kp in kps])
        kps_ints = np.int32(kps_floats)
        in_img_mask = kps_ints[:, 0] >= 0
        in_img_mask = np.logical_and(in_img_mask, kps_ints[:, 0] < img.shape[1])
        in_img_mask = np.logical_and(in_img_mask, kps_ints[:, 1] >= 0)
        in_img_mask = np.logical_and(in_img_mask, kps_ints[:, 1] < img.shape[0])
        kps_ints = kps_ints[in_img_mask]
        kps = [kp for i, kp in enumerate(kps) if in_img_mask[i]]
        descs = descs[in_img_mask]

        valid_keys_set = set(valid_components_dict.keys())
        all_indices_set = set(range(np.max(components_indices) + 1))
        non_valid_indices = list(all_indices_set - valid_keys_set)

        filter_non_valid = np.zeros(kps_ints.shape[0])
        for non_valid_index in non_valid_indices:
            filter_non_valid = np.logical_or(filter_non_valid, components_indices[kps_ints[:, 1], kps_ints[:, 0]] == non_valid_index)

        kps = [kp for i, kp in enumerate(kps) if filter_non_valid[i]]
        descs = descs[filter_non_valid]

    all_kps.extend(kps)
    unrectified_indices = np.zeros(len(all_kps), dtype=bool)
    unrectified_indices[-len(kps):] = True

    if all_descs is None:
        all_descs = descs
    else:
        all_descs = np.vstack((all_descs, descs))

    if show or save:
        no_component_img = img.copy()
        cv.drawKeypoints(no_component_img, kps, no_component_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.title("{} - no valid component".format(img_name))
        plt.imshow(no_component_img)
        if save:
            plt.savefig("{}_rectified_no_valid_component".format(out_prefix))
        show_or_close(show)

        all_img = img.copy()
        cv.drawKeypoints(all_img, all_kps, all_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.title("All keypoints")
        plt.imshow(all_img)
        if save:
            plt.savefig("{}_rectified_all".format(out_prefix))
        show_or_close(show)
        print("{} keypoints found".format(len(all_kps)))

    return all_kps, all_descs, unrectified_indices


def possibly_upsample_normals(img, normal_indices):

    if img.shape[0] != normal_indices.shape[0]:
        epsilon = 0.003
        hard_epsilon = 0.1
        aspect_ratio_diff = abs(img.shape[0] / normal_indices.shape[0] - img.shape[1] / normal_indices.shape[1])
        if aspect_ratio_diff >= hard_epsilon:
            raise Exception("{} and {} not of the same aspect ratio".format(normal_indices.shape, img.shape))
        else:
            if img.shape[0] < normal_indices.shape[0]:
                print("WARNING: img.shape[0] < normal_indices.shape[0]")
            if aspect_ratio_diff >= epsilon:
                print("WARNING: {} and {} not of the same aspect ratio".format(normal_indices.shape, img.shape))
            print("Will upsample the normals")
            normal_indices = resample_nearest_numpy(normal_indices, img.shape[0], img.shape[1])

    return normal_indices


