from pathlib import Path

import matplotlib.pyplot as plt

import clustering
from img_utils import show_and_save_normal_clusters_3d, show_point_cloud, show_or_close, get_degrees_between_normals
from utils import *


def get_gauss_weighted_coeffs_for_window(window_size=5, sigma=1.33, device=torch.device('cpu')):

    x = torch.linspace(-float(window_size//2), float(window_size//2), window_size).to(device)
    x, y = torch.meshgrid(x, x)

    normalizing_gauss_coeffs = 1.0 / (2.0 * math.pi * sigma ** 2)
    gauss_coeffs = normalizing_gauss_coeffs * torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma**2))

    gauss_weighted_coeffs = gauss_coeffs.flatten()
    gauss_weighted_coeffs_normalized = window_size ** 2 * gauss_weighted_coeffs / gauss_weighted_coeffs.sum()

    assert math.fabs(gauss_weighted_coeffs_normalized.sum() - window_size ** 2) < 0.0001

    return gauss_weighted_coeffs_normalized


def show_or_save_clusters(normals, normal_indices_np, cluster_repr_normal_np, out_dir, img_name, show=False, save=False):

    if show or save:
        img = np.ndarray(normal_indices_np.shape + (3,))
        img[:, :, 0][normal_indices_np == 0] = 255
        img[:, :, 0][normal_indices_np != 0] = 0
        img[:, :, 1][normal_indices_np == 1] = 255
        img[:, :, 1][normal_indices_np != 1] = 0
        img[:, :, 2][normal_indices_np == 2] = 255
        img[:, :, 2][normal_indices_np != 2] = 0

        plt.figure(figsize=(9, 9))
        color_names = ["red", "green", "blue"]
        title = "{}:\n".format(img_name)
        np.set_printoptions(suppress=True, precision=3)
        for i in range(cluster_repr_normal_np.shape[0]):
            degrees_z = np.array([math.acos(np.dot(np.array([0, 0, -1]), cluster_repr_normal_np[i])) * 180 / math.pi])
            title = "{}{}={} - {} deg.,\n".format(title, color_names[i], cluster_repr_normal_np[i], degrees_z)

        degrees_list = get_degrees_between_normals(cluster_repr_normal_np)
        if len(degrees_list) > 0:
            title = title + "\n degrees between pairs of normals:\n {}".format(",\n".join([str(s) for s in degrees_list]))
        plt.title(title)
        plt.imshow(img)
        if save:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = '{}/{}'.format(out_dir, img_name[:-4])
            plt.savefig("{}_clusters.jpg".format(out_path))
        show_or_close(show)

    if save:
        cv.imwrite("{}_clusters_indices_unused.png".format(out_path), normal_indices_np)
        np.savetxt('{}_clusters_normals_unused.txt'.format(out_path), cluster_repr_normal_np, delimiter=',', fmt='%1.8f')

    if show:
        show_and_save_normal_clusters_3d(normals, cluster_repr_normal_np, normal_indices_np, show, save, out_dir, img_name)


def cluster_normals(normals, filter_mask=None, mean_shift_type=None, adaptive=False, return_all=False, device=torch.device("cpu"), handle_antipodal_points=True):

    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)
        raise Exception("should not happen")

    if filter_mask is None:
        # only ones
        filter_mask = torch.ones(normals.shape[:2]).to(device)
    elif isinstance(filter_mask, np.ndarray):
        filter_mask = torch.from_numpy(filter_mask).to(device)

    Timer.start_check_point("clustering normals")
    cluster_repr_normal, normal_indices, valid_clusters = clustering.cluster(normals, filter_mask, mean_shift_type, adaptive, return_all, device=device, handle_antipodal_points=handle_antipodal_points)

    print("cluster_repr_normal.device: {}".format(cluster_repr_normal.device))
    print("normal_indices.device: {}".format(normal_indices.device))

    normal_indices_np = normal_indices.detach().cpu().numpy().astype(dtype=np.uint8)
    cluster_repr_normal_np = cluster_repr_normal.detach().cpu().numpy()

    Timer.end_check_point("clustering normals")

    return cluster_repr_normal_np, normal_indices_np, valid_clusters


def get_file_names_from_dir(input_dir: str, limit: int, interesting_files: list, suffix: str):
    if interesting_files is not None:
        return interesting_files
    else:
        return get_file_names(input_dir, suffix, limit)


def show_sky_mask(img, filter_mask, img_name, show, save=False, path=None, title=None):
    if not save and not show:
        return
    fig = plt.figure(figsize=(9, 9))
    if title is None:
        title = "sky mask"
    plt.title("{} for {}".format(title, img_name))
    plt.axis('off')
    ax = fig.add_subplot(121)
    ax.imshow(img)
    ax = fig.add_subplot(122)
    ax.imshow(filter_mask)
    if save:
        plt.savefig(path)
    show_or_close(show)


def compute_only_normals(
        focal_length,
        orig_height,
        orig_width,
        depth_data_read_directory,
        depth_data_file_name,
        simple_weighing=True,
        smaller_window=False,
        device=torch.device('cpu')):

    depth_data = read_depth_data(depth_data_file_name, depth_data_read_directory)
    normals, s_values = compute_normals_from_svd(focal_length, orig_height, orig_width, depth_data, simple_weighing, smaller_window, device=device)
    return normals, s_values


def compute_normals_from_svd(
        focal_length,
        orig_height,
        orig_width,
        depth_data,
        simple_weighing=True,
        smaller_window=False,
        device=torch.device('cpu'),
        svd_weighted=True,
        svd_weighted_sigma=0.8,
):

    window_size = 5

    depth_height = depth_data.shape[2]
    depth_width = depth_data.shape[3]

    # depth_data shapes
    f_factor_x = depth_width / orig_width
    f_factor_y = depth_height / orig_height
    if abs(f_factor_y - f_factor_x) > 0.001:
        print("WARNING: downsampled anisotropically")
    real_focal_length_x = focal_length * f_factor_x
    real_focal_length_y = focal_length * f_factor_y

    # this is just to avoid handling odd numbers for linspace (see below)
    # this should be safe as the depth maps are 512 x n*32
    assert depth_height % 2 == 0
    assert depth_width % 2 == 0

    # NOTE this can be done only once #performance
    width_linspace = torch.linspace(-depth_width/2, depth_width/2 - 1, steps=depth_width).to(device)
    height_linspace = torch.linspace(-depth_height/2, depth_height/2 - 1, steps=depth_height).to(device)

    grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

    origin_to_z1 = torch.sqrt(1 + (grid_x / real_focal_length_x) ** 2 + (grid_y / real_focal_length_y) ** 2).to(device)

    # (1, h, w, 3)
    point_cloud = torch.Tensor(depth_data.shape[1:] + (3,)).to(device)
    point_cloud[:, :, :, 2] = depth_data[0] / origin_to_z1
    point_cloud[:, :, :, 0] = point_cloud[:, :, :, 2] * grid_x / real_focal_length_x
    point_cloud[:, :, :, 1] = point_cloud[:, :, :, 2] * grid_y / real_focal_length_y

    show = False
    if show:
        x = point_cloud[0, -100:, -100:, 0].flatten()
        y = point_cloud[0, -100:, -100:, 1].flatten()
        z = point_cloud[0, -100:, -100:, 2].flatten()
        show_point_cloud(x, y, z)

    # (1, h, w, 3) -> (3, 1, h, w)
    point_cloud = point_cloud.permute(3, 0, 1, 2)

    unfold = torch.nn.Unfold(kernel_size=(window_size, window_size))

    # (3, window_size ** 2, (h - window_size//2) * (w - window_size//2))
    unfolded = unfold(point_cloud)

    new_depth_height = depth_height - (window_size//2 * 2)
    new_depth_width = depth_width - (window_size//2 * 2)
    assert unfolded.shape[2] == new_depth_height * new_depth_width

    window_pixels = window_size ** 2
    centered = unfolded - (torch.sum(unfolded, dim=1) / window_pixels).unsqueeze(dim=1)

    # (-1, -1, -1) -> ((h - window_size // 2) * (w - window_size // 2), window_size ** 2, 3)
    centered = centered.permute(2, 1, 0)

    if svd_weighted:
        w_diag = torch.diag_embed(get_gauss_weighted_coeffs_for_window(window_size=window_size, sigma=svd_weighted_sigma))
        w_diag = w_diag.to(device)
        c2 = w_diag @ centered
        U, s_values, V = torch.svd(c2)
    else:
        U, s_values, V = torch.svd(centered)

    normals = V[:, :, 2]
    normals = normals.reshape(new_depth_height, new_depth_width, 3)

    # flip if z > 0
    where = torch.where(normals[:, :, 2] > 0)
    normals[where[0], where[1]] = -normals[where[0], where[1]]

    # is this necessary?
    normals = normals / torch.norm(normals, dim=2).unsqueeze(dim=2)

    normals = pad_normals(normals, window_size=window_size)
    assert normals.shape[0] == depth_height
    assert normals.shape[1] == depth_width

    s_values = s_values.reshape(new_depth_height, new_depth_width, 3)
    s_values = pad_normals(s_values, window_size=window_size)
    assert s_values.shape[0] == depth_height
    assert s_values.shape[1] == depth_width

    return normals, s_values


