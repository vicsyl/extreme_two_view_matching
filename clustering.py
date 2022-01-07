import math

import torch
import torch.nn.functional as F

from utils import Timer


def assert_almost_equal(one, two):
    assert math.fabs(one - two) < 0.000001


def recompute_points_threshold_ratio(angle_distance_threshold_degrees, points_threshold_ratio_factor=1.0):
    return 0.13 * (angle_distance_threshold_degrees / 30) * points_threshold_ratio_factor


def from_degrees_to_dist(degrees, log_key, factor=1.0):
    rads = factor * degrees * math.pi / 180
    distance = math.sin(rads / 2) * 2
    print("{}: degrees: {}, distance: {}".format(log_key, degrees, distance))
    return distance


class Clustering:

    # primary params
    N_points = 300
    angle_distance_threshold_degrees = 35
    distance_threshold = from_degrees_to_dist(angle_distance_threshold_degrees, "bin angle")
    distance_inter_cluster_threshold_factor = 2.5
    distance_inter_cluster_threshold = from_degrees_to_dist(angle_distance_threshold_degrees, "seed inter cluster angle", distance_inter_cluster_threshold_factor)

    ms_kernel_max_distance = distance_threshold
    ms_adjustment_th = 0.1
    ms_max_iter = 100
    ms_bandwidth = ms_kernel_max_distance / 2
    ms_distance_inter_cluster_threshold_factor = 2
    ms_distance_inter_cluster_threshold = from_degrees_to_dist(angle_distance_threshold_degrees, "mean shift seed inter cluster angle", ms_distance_inter_cluster_threshold_factor)

    # previous hard-coded value: 0.13
    points_threshold_ratio = recompute_points_threshold_ratio(angle_distance_threshold_degrees, points_threshold_ratio_factor=1.0)

    @staticmethod
    def recompute(points_threshold_ratio_factor):

        Clustering.distance_threshold = from_degrees_to_dist(Clustering.angle_distance_threshold_degrees, "bin angle")
        Clustering.distance_inter_cluster_threshold = from_degrees_to_dist(Clustering.angle_distance_threshold_degrees, "seed inter cluster angle", Clustering.distance_inter_cluster_threshold_factor)
        Clustering.ms_kernel_max_distance = Clustering.distance_threshold
        Clustering.ms_bandwidth = Clustering.ms_kernel_max_distance / 2
        Clustering.ms_distance_inter_cluster_threshold = from_degrees_to_dist(Clustering.angle_distance_threshold_degrees, "mean shift seed inter cluster angle", Clustering.ms_distance_inter_cluster_threshold_factor)

        Clustering.points_threshold_ratio = recompute_points_threshold_ratio(Clustering.angle_distance_threshold_degrees, points_threshold_ratio_factor)
        print("Recomputed")
        Clustering.log()

    @staticmethod
    def log():
        print("Clustering:")
        print("\tN_points\t{}".format(Clustering.N_points))
        print("\tangle_distance_threshold\t{} degrees".format(Clustering.angle_distance_threshold_degrees))
        print("\tdistance_threshold\t{}".format(Clustering.distance_threshold))
        print("\tdistance_inter_cluster_threshold_factor\t{}".format(Clustering.distance_inter_cluster_threshold_factor))
        print("\tdistance_inter_cluster_threshold\t{}".format(Clustering.distance_inter_cluster_threshold))
        print("\tms_kernel_max_distance\t{}".format(Clustering.ms_kernel_max_distance))
        print("\tms_adjustment_th\t{}".format(Clustering.ms_adjustment_th))
        print("\tms_max_iter\t{}".format(Clustering.ms_max_iter))
        print("\tms_bandwidth\t{}".format(Clustering.ms_bandwidth))
        print("\tms_distance_inter_cluster_threshold_factor\t{}".format(Clustering.ms_distance_inter_cluster_threshold_factor))
        print("\tms_distance_inter_cluster_threshold\t{}".format(Clustering.ms_distance_inter_cluster_threshold))
        print("\tpoints_threshold_ratio\t{}".format(Clustering.points_threshold_ratio))


# https://web.archive.org/web/20120107030109/http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere#Spirals
def n_points_across_half_sphere(N):
    """
    :param N: number of points to distribute across a hemisphere
    :return:
    """
    s = 3.6 / math.sqrt(N)
    dz = 1.0 / N
    longitude = 0
    z = 1 - dz / 2

    points = torch.zeros((N, 3))
    for k in range(N):
        r = math.sqrt(1 - z * z)
        points[k] = torch.tensor([math.cos(longitude) * r, math.sin(longitude) * r, -z])
        z = z - dz
        longitude = longitude + s / r

    return points


def angle_2_unit_vectors(v1, v2):
    arg = torch.clamp(v1.T @ v2, min=-1.0, max=1.0)
    return math.acos(arg) / math.pi * 180


def bilateral_filter(normals: torch.Tensor, filter_mask, filter_range=9, spatial_sigma=10.0, normal_sigma=3.0, device=torch.device("cpu")):
    """
    :param normals: Tensor(h, w, 3)
    :param filter_mask: (h, w)
    :param filter_range: int
    :param spatial_sigma: float
    :param normal_range: int
    :param normal_sigma: float
    :return:
    """
    normals = normals.permute(2, 0, 1)

    weights = torch.zeros((filter_range, filter_range) + normals.shape[1:])

    # think about reordering the dimensions
    padded_normals = F.pad(normals.unsqueeze(0), (filter_range//2, filter_range//2, filter_range//2, filter_range//2), mode='replicate')[0]

    for s_1 in range(filter_range):
        for s_2 in range(filter_range):
            offset_1 = s_1 - filter_range // 2
            offset_2 = s_2 - filter_range // 2
            spatial_diff = offset_1 ** 2 + offset_2 ** 2
            spatial_diff = -spatial_diff / (2 * spatial_sigma ** 2)
            # weights = conv(c1, spatial_filter)

            norm_diff = padded_normals[:, s_1:s_1 + normals.shape[1], s_2:s_2 + normals.shape[2]] - normals
            norm_diff = torch.norm(norm_diff, dim=0)
            norm_diff = -norm_diff / (2 * normal_sigma ** 2)
            weights[s_1, s_2] = torch.exp(norm_diff + spatial_diff)

    filtered = torch.zeros_like(normals).to(device)
    print("filtered device: {}".format(filtered.device))
    print("weights device: {}".format(weights.device))
    print("padded_normals device: {}".format(padded_normals.device))
    for s_1 in range(filter_range):
        for s_2 in range(filter_range):
            filtered = filtered + padded_normals[:, s_1:s_1 + normals.shape[1], s_2:s_2 + normals.shape[2]] * weights[s_1, s_2]

    weights_sum = weights.sum(dim=(0, 1))
    torch.clamp(weights_sum, min=1e-19)

    filtered = filtered / weights_sum
    filtered = filtered * filter_mask

    filtered = filtered.permute(1, 2, 0)

    return filtered


def cluster(normals: torch.Tensor,
            filter_mask,
            mean_shift_type=None,
            adaptive=False,
            return_all=False,
            device=torch.device("cpu"),
            handle_antipodal_points=True
            ):
    """
    :param normals: Tensor(h, w, 3)
    :param filter_mask: (h, w)
    :param mean_shift_type: boolean - if mean or mean-shift should be run after the binning (None/"full"/"mean")
    :return: normals: (n, 3), cluster_membership (h, w) - index of cluster or 3 for no cluster
    """

    points_threshold = torch.prod(torch.tensor(normals.shape[:2])) * Clustering.points_threshold_ratio
    points_threshold = points_threshold.to(device)

    timer_label = "clustering for N={}".format(Clustering.N_points)
    Timer.start_check_point(timer_label)
    n_centers = n_points_across_half_sphere(Clustering.N_points).to(device)

    n_centers = n_centers.expand(normals.shape[0], normals.shape[1], -1, -1)
    n_centers = n_centers.permute(2, 0, 1, 3)

    # handle all centers at once
    if handle_antipodal_points:
        one = (n_centers - normals)[None]
        two = (n_centers + normals)[None]
        diffs = torch.vstack((one, two))
        diff_norm = torch.norm(diffs, dim=4)
        mins_args = torch.min(diff_norm, dim=0)
        diff_norm = mins_args[0]
        argmin_factor = 1 - 2 * mins_args[1]
    else:
        diffs = n_centers - normals
        diff_norm = torch.norm(diffs, dim=3)

    near_ones_per_cluster_center = torch.where(diff_norm < Clustering.distance_threshold, 1, 0)
    near_ones_per_cluster_center = torch.logical_and(near_ones_per_cluster_center, filter_mask)

    sums = near_ones_per_cluster_center.sum(dim=(1, 2))

    sortd = torch.sort(sums, descending=True)

    cluster_centers = []

    arg_mins = torch.ones(normals.shape[:2]).to(device) * 3
    arg_mins = arg_mins.to(torch.int)

    max_clusters = 3
    valid_clusters = max_clusters
    for center_index, points in zip(sortd[1], sortd[0]):
        if len(cluster_centers) >= max_clusters:
            break
        if points < points_threshold:
            if return_all:
                valid_clusters = len(cluster_centers)
            else:
                break

        def is_distance_ok(new_center, threshold):
            for cluster_center in cluster_centers:
                if handle_antipodal_points:
                    one = (new_center - cluster_center)[None]
                    two = (new_center + cluster_center)[None]
                    diff_new = torch.vstack((one, two))
                    diff_norm = torch.norm(diff_new, dim=1).min()
                else:
                    diff = new_center - cluster_center
                    diff_norm = torch.norm(diff)
                if diff_norm < threshold:
                    return False
            return True

        if mean_shift_type == "full":
            th = Clustering.ms_distance_inter_cluster_threshold
        else:
            th = Clustering.distance_inter_cluster_threshold

        distance_ok = is_distance_ok(n_centers[center_index, 0, 0], th)

        if distance_ok:
            if mean_shift_type is None:
                cluster_center = n_centers[center_index, 0, 0]
                cluster_centers.append(cluster_center)
                arg_mins[near_ones_per_cluster_center[center_index]] = len(cluster_centers) - 1
            elif mean_shift_type == "mean":
                # near_ones_per_cluster_center needs recomputing
                coords = torch.where(near_ones_per_cluster_center[center_index, :, :])
                if handle_antipodal_points:
                    normals_to_mean = normals * argmin_factor[center_index].unsqueeze(2)
                    normals_to_mean = normals_to_mean[coords[0], coords[1]]
                else:
                    normals_to_mean = normals[coords[0], coords[1]]

                # normals_to_mean conv with kernel + normalization (to norm == 1)
                cluster_center = normals_to_mean.sum(dim=0) / normals_to_mean.shape[0]
                cluster_center = cluster_center / torch.norm(cluster_center)
                cluster_centers.append(cluster_center)

                # just printing the shift
                angle = angle_2_unit_vectors(cluster_center, n_centers[center_index, 0, 0])
                print("delta (mean vs. cluster center): {} degrees".format(angle))

                # recompute the neighborhood just for the new center
                if handle_antipodal_points:
                    one = (cluster_center - normals)[None]
                    two = (cluster_center + normals)[None]
                    diffs = torch.vstack((one, two))
                    distances_new = torch.norm(diffs, dim=3)
                    distances_new = torch.min(distances_new, dim=0)[0]
                    neighborhood = torch.where(distances_new < Clustering.distance_threshold, 1, 0)
                    neighborhood = torch.logical_and(neighborhood, filter_mask)
                else:
                    distances = torch.norm(cluster_center - normals, dim=2) #.expand(normals.shape[0], normals.shape[1])
                    neighborhood = torch.where(distances < Clustering.distance_threshold, 1, 0)
                    neighborhood = torch.logical_and(neighborhood, filter_mask)

                coords = torch.where(neighborhood)
                arg_mins[coords[0], coords[1]] = len(cluster_centers) - 1

            elif mean_shift_type == "full":

                if handle_antipodal_points:
                    raise Exception("handle_antipodal_points not implemented for full mean shift")

                cluster_center = n_centers[center_index, 0, 0]
                orig_center = cluster_center

                angle_diff = Clustering.ms_adjustment_th
                for cluster_iter in range(Clustering.ms_max_iter):
                    if angle_diff < Clustering.ms_adjustment_th:
                        break

                    distances = torch.norm(cluster_center - normals, dim=2).expand(normals.shape[0], normals.shape[1])

                    neighborhood = torch.where(distances < Clustering.ms_kernel_max_distance, 1, 0)
                    neighborhood = torch.logical_and(neighborhood, filter_mask)
                    print("{} data points in the bin - {}".format(neighborhood.sum(), cluster_center))

                    coords = torch.where(neighborhood)
                    normals_for_shift = normals[coords[0], coords[1]]
                    distances_squared = (distances[coords[0], coords[1]] / Clustering.ms_bandwidth) ** 2

                    def normal_kernel_values(distances_sq):
                        return torch.exp(distances_sq * -0.5) * 0.5 / math.pi

                    def uniform_kernel_values(distances_sq):
                        return torch.ones_like(distances_sq).to(device)

                    # normalization const. ignored (should be very close to 1 anyway)
                    kernel_values = uniform_kernel_values(distances_squared)
                    kernel_conv_value = kernel_values.sum()
                    print("kernel_conv_value: {}".format(kernel_conv_value))
                    new_center = (normals_for_shift * kernel_values.expand(3, -1).permute(1, 0)).sum(dim=0) / kernel_conv_value
                    new_center = new_center / torch.norm(new_center)

                    if adaptive:

                        def get_neighborhood_new():
                            distances_new = torch.norm(new_center - normals, dim=2).expand(normals.shape[0],
                                                                                           normals.shape[1])
                            neighborhood_new = torch.where(distances_new < Clustering.ms_kernel_max_distance, 1, 0)
                            neighborhood_new = torch.logical_and(neighborhood_new, filter_mask)
                            return neighborhood_new

                        neighborhood_new = get_neighborhood_new()
                        while neighborhood_new.sum() < neighborhood.sum():
                            print("fewer points in the vicinity, stopping, slowing down")
                            new_center = cluster_center * 0.5 + new_center * 0.5
                            new_center = new_center / torch.norm(new_center)
                            angle_diff = angle_2_unit_vectors(cluster_center, new_center)
                            if angle_diff < Clustering.ms_adjustment_th:
                                break
                            else:
                                print("angle: {}, new_center: {}".format(angle_diff, new_center))
                            neighborhood_new = get_neighborhood_new()

                        if neighborhood_new.sum() < neighborhood.sum():
                            cluster_iter = Clustering.ms_max_iter
                            break
                        else:
                            print("adaptive step helped")

                    angle_diff = angle_2_unit_vectors(cluster_center, new_center)
                    print("mode adjustment (iteration): {} degrees".format(angle_diff))
                    angle_diff_overall = angle_2_unit_vectors(orig_center, new_center)
                    print("mode adjustment (overall): {} degrees".format(angle_diff_overall))
                    print("orig: {}, old: {}, new: {}".format(orig_center, cluster_center, new_center))

                    cluster_center = new_center

                distance_ok = is_distance_ok(cluster_center, Clustering.distance_inter_cluster_threshold)
                if distance_ok:
                    cluster_centers.append(cluster_center)
                    distances = torch.norm(cluster_center - normals, dim=2).expand(normals.shape[0], normals.shape[1])
                    neighborhood = torch.where(distances < Clustering.distance_threshold, 1, 0)
                    neighborhood = torch.logical_and(neighborhood, filter_mask)
                    print("at the end {} data points in the bin - {}".format(neighborhood.sum(), cluster_center))

                    distances_2 = torch.norm(cluster_center - normals, dim=2).expand(normals.shape[0], normals.shape[1])
                    neighborhood_2 = torch.where(distances_2 < Clustering.ms_kernel_max_distance, 1, 0)
                    neighborhood_2 = torch.logical_and(neighborhood_2, filter_mask)
                    print("{} data points in the bin - {}".format(neighborhood_2.sum(), cluster_center))

                    coords = torch.where(neighborhood)
                    arg_mins[coords[0], coords[1]] = len(cluster_centers) - 1

    if len(cluster_centers) == 1:
        cluster_centers = cluster_centers[0]
        cluster_centers = torch.unsqueeze(cluster_centers, dim=0)
    elif len(cluster_centers) > 1:
        cluster_centers = torch.vstack(cluster_centers)
    else:
        # NOTE corner case - no clusters found
        cluster_centers = torch.zeros((0, 3))

    Timer.end_check_point(timer_label)

    valid_clusters = min(valid_clusters, len(cluster_centers))
    return cluster_centers, arg_mins, valid_clusters
