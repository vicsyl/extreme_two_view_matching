import sys
import traceback
from typing import List

from scene_info import *
from utils import *

"""
DISCLAIMER: the following methods have been adopted from https://github.com/ducha-aiki/ransac-tutorial-2020-data:
- normalize_keypoints
- quaternion_from_matrix
- evaluate_R_t
"""


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()
        raise

    return err_q, err_t


def evaulate_R_t_safe(dR, dt, R, t):

    try:
        err_q, err_t = evaluate_R_t(dR, dt, R, t)
    except:
        print("WARNING: evaulate_R_t_safe")
        print(traceback.format_exc(), file=sys.stdout)

        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv.recoverPose(E, p1n, p2n)
        err_q, err_t = evaulate_R_t_safe(dR, dt, R, t)

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t, R


def get_GT_R_t(img_pair: ImagePairEntry, scene_info: SceneInfo):

    img_entry_1: ImageEntry = scene_info.img_info_map[img_pair.img1]
    T1 = img_entry_1.t
    R1 = img_entry_1.R

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = img_entry_2.t
    R2 = img_entry_2.R

    dR = R2 @ R1.T
    dt = T2 - dR @ T1

    return dR, dt


def compare_R_to_GT(img_pair: ImagePairEntry, scene_info: SceneInfo, r):
    dR, dt = get_GT_R_t(img_pair, scene_info)
    err_q, _ = evaulate_R_t_safe(dR, dt, r, dt)
    return err_q


def compare_poses(E, img_pair: ImagePairEntry, scene_info: SceneInfo, pts1, pts2, img_data_list):

    img_entry_1: ImageEntry = scene_info.img_info_map[img_pair.img1]
    T1 = img_entry_1.t
    R1 = img_entry_1.R

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = img_entry_2.t
    R2 = img_entry_2.R

    dR = R2 @ R1.T
    dt = T2 - dR @ T1

    K1 = scene_info.get_img_K(img_pair.img1, img_data_list[0].img)
    K2 = scene_info.get_img_K(img_pair.img2, img_data_list[1].img)

    p1n = normalize_keypoints(pts1, K1).astype(np.float64)
    p2n = normalize_keypoints(pts2, K2).astype(np.float64)

    err_q, err_t, dr_est = eval_essential_matrix(p1n, p2n, E, dR, dt)

    rot_vec_deg_est = get_rot_vec_deg(dr_est)
    rot_vec_deg_gt = get_rot_vec_deg(dR)

    print("rotation vector(GT): {}".format(rot_vec_deg_gt))
    print("rotation vector(est): {}".format(rot_vec_deg_est))
    print("errors (R, T): ({} degrees, {} (unscaled))".format(np.rad2deg(err_q), err_t))

    return err_q, err_t


# a HACK that enables pickling of cv2.KeyPoint - see
# https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror/48832618
import copyreg
import cv2


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


@dataclass
class ImageData:
    img: np.ndarray
    key_points: List[cv.KeyPoint]
    descriptions: object
    real_K: np.ndarray
    normals: np.ndarray
    components_indices: np.ndarray
    valid_components_dict: dict

    @staticmethod
    def from_serialized_data(img, real_K, img_serialized_data):
        return ImageData(img=img,
                         real_K=real_K,
                         key_points=img_serialized_data.kpts,
                         descriptions=img_serialized_data.descs,
                         normals=img_serialized_data.normals,
                         components_indices=img_serialized_data.components_indices,
                         valid_components_dict=img_serialized_data.valid_components_dict)

    def to_serialized_data(self):
        return ImageSerializedData(kpts=self.key_points,
                                   descs=self.descriptions,
                                   normals=self.normals,
                                   components_indices=self.components_indices,
                                   valid_components_dict=self.valid_components_dict)


@dataclass
class ImageSerializedData:
    kpts: list
    descs: list
    normals: np.ndarray
    components_indices: np.ndarray
    valid_components_dict: dict


@dataclass
class Stats:

    inliers_against_gt: (int, int, int)
    tentatives_1: (float, float)
    tentatives_2: (float, float)
    error_R: float
    error_T: float
    tentative_matches: int
    inliers: int
    all_features_1: int
    all_features_2: int
    E: np.ndarray
    normals1: np.ndarray
    normals2: np.ndarray

    # legacy
    def make_brief(self):
        self.src_pts_inliers = None
        self.dst_pts_inliers = None
        self.src_tentatives = None
        self.dst_tentatives = None
        self.kpts1 = None
        self.kpts2 = None


def evaluate_matching(scene_info,
                      E,
                      img_data_list,
                      tentative_matches,
                      inlier_mask,
                      img_pair,
                      stats_map,
                      ransac_th,
                      ):

    print("Image pair: {} <-> {}:".format(img_pair.img1, img_pair.img2))
    print("Number of inliers: {}".format(inlier_mask[inlier_mask == [1]].shape[0]))
    print("Number of outliers: {}".format(inlier_mask[inlier_mask == [0]].shape[0]))

    src_tentatives_2d, dst_tentatives_2d = split_points(tentative_matches, img_data_list[0].key_points, img_data_list[1].key_points)
    src_pts_inliers = src_tentatives_2d[inlier_mask[:, 0] == [1]]
    dst_pts_inliers = dst_tentatives_2d[inlier_mask[:, 0] == [1]]

    error_R, error_T = compare_poses(E, img_pair, scene_info, src_pts_inliers, dst_pts_inliers, img_data_list)
    inliers = np.sum(np.where(inlier_mask[:, 0] == [1], 1, 0))

    if is_rectified_condition(img_data_list[0]):
        _, unique, counts = get_normals_stats(img_data_list, src_tentatives_2d, dst_tentatives_2d)
        print("Matching stats in evaluation:")
        print("unique plane correspondence counts of tentatives:\n{}".format(np.vstack((unique.T, counts)).T))

    count_sampson_gt, \
    count_symmetrical_gt, \
    count_sampson_estimated, \
    count_symmetrical_estimated = evaluate_tentatives_agains_ground_truth(scene_info,
                                                                          img_pair,
                                                                          img_data_list,
                                                                          src_tentatives_2d,
                                                                          dst_tentatives_2d,
                                                                          [ransac_th, 0.1, 0.5, 1, 3],
                                                                          E, inliers)

    stats = Stats(inliers_against_gt=count_symmetrical_gt,
                  tentatives_1=src_tentatives_2d,
                  tentatives_2=dst_tentatives_2d,
                  error_R=error_R,
                  error_T=error_T,
                  tentative_matches=len(tentative_matches),
                  inliers=inliers,
                  all_features_1=len(img_data_list[0].key_points),
                  all_features_2=len(img_data_list[1].key_points),
                  E=E,
                  normals1=img_data_list[0].normals,
                  normals2=img_data_list[1].normals,
                  )

    key = "{}_{}".format(img_pair.img1, img_pair.img2)
    stats_map[key] = stats
    return stats


def print_stats(stat_name: str, stat_in_list: list):
    np_ar = np.array(stat_in_list)
    print("average {}: {}".format(stat_name, np.sum(np_ar) / len(stat_in_list)))


def vector_product_matrix(vec: np.ndarray):
    return np.array([
        [    0.0, -vec[2],  vec[1]],
        [ vec[2],     0.0, -vec[0]],
        [-vec[1],  vec[0],     0.0],
    ])


def evaluate_tentatives_agains_ground_truth(scene_info: SceneInfo,
                                            img_pair: ImagePairEntry,
                                            img_data_list,
                                            src_tentatives_2d,
                                            dst_tentatives_2d,
                                            thresholds,
                                            est_E,
                                            inliers_from_ransac):

    def get_T_R_K_inv(img_key, img):
        img_entry: ImageEntry = scene_info.img_info_map[img_key]
        T = np.array(img_entry.t)
        R = img_entry.R
        K = scene_info.get_img_K(img_key, img)
        K_inv = np.linalg.inv(K)
        return T, R, K_inv, K

    T1, R1, K1_inv, K1 = get_T_R_K_inv(img_pair.img1, img_data_list[0].img)
    src_tentative_h = np.ndarray((src_tentatives_2d.shape[0], 3))
    src_tentative_h[:, :2] = src_tentatives_2d
    src_tentative_h[:, 2] = 1.0

    T2, R2, K2_inv, K2 = get_T_R_K_inv(img_pair.img2, img_data_list[1].img)
    dst_tentative_h = np.ndarray((dst_tentatives_2d.shape[0], 3))
    dst_tentative_h[:, :2] = dst_tentatives_2d
    dst_tentative_h[:, 2] = 1.0

    F_ground_truth = K2_inv.T @ R2 @ vector_product_matrix(T2 - T1) @ R1.T @ K1_inv
    F_ground_truth = torch.from_numpy(F_ground_truth).unsqueeze(0)

    src_pts_torch = torch.from_numpy(src_tentatives_2d).type(torch.DoubleTensor)
    dst_pts_torch = torch.from_numpy(dst_tentatives_2d).type(torch.DoubleTensor)

    kornia_sampson_gt = KG.epipolar.sampson_epipolar_distance(src_pts_torch, dst_pts_torch, F_ground_truth).numpy()
    kornia_symmetrical_gt = KG.epipolar.symmetrical_epipolar_distance(src_pts_torch, dst_pts_torch, F_ground_truth).numpy()

    computed_F = KG.fundamental_from_essential(torch.from_numpy(est_E), torch.from_numpy(K1), torch.from_numpy(K2))
    computed_F = computed_F.unsqueeze(0)

    kornia_sampson_estimated = KG.epipolar.sampson_epipolar_distance(src_pts_torch, dst_pts_torch, computed_F).numpy()
    kornia_symmetrical_estimated = KG.epipolar.symmetrical_epipolar_distance(src_pts_torch, dst_pts_torch, computed_F).numpy()

    count_sampson_gt = np.zeros(len(thresholds), dtype=int)
    count_symmetrical_gt = np.zeros(len(thresholds), dtype=int)
    count_sampson_estimated = np.zeros(len(thresholds), dtype=int)
    count_symmetrical_estimated = np.zeros(len(thresholds), dtype=int)

    print("Matching stats in inliers:")
    def evaluate_metric(metric, th, label):
        mask = (np.abs(metric) < th)[0]
        if is_rectified_condition(img_data_list[0]):
            _, unique, counts = get_normals_stats(img_data_list, src_tentatives_2d, dst_tentatives_2d, mask)
            print("{} < {}:".format(label, th))
            print("unique plane correspondence counts:\n{}".format(np.vstack((unique.T, counts)).T))
        return np.sum(mask)

    for i in range(len(thresholds)):
        count_sampson_gt[i] = evaluate_metric(kornia_sampson_gt, thresholds[i], "sampson gt")
        count_symmetrical_gt[i] = evaluate_metric(kornia_symmetrical_gt, thresholds[i], "symmetrical gt")
        count_sampson_estimated[i] = evaluate_metric(kornia_sampson_estimated, thresholds[i], "sampson estimated")
        count_symmetrical_estimated[i] = evaluate_metric(kornia_symmetrical_estimated, thresholds[i], "symmetrical estimated")

    print("inliers from ransac: {}".format(inliers_from_ransac))
    print("thresholds for inliers: {}".format(thresholds))
    print("inliers against GT - sampson:{}".format(count_sampson_gt))
    print("inliers against GT - symmetrical:{}".format(count_symmetrical_gt))
    print("inliers against estimated E/F - sampson:{}".format(count_sampson_estimated))
    print("inliers against estimated E/F - symmetrical:{}".format(count_symmetrical_estimated))

    return count_sampson_gt, count_symmetrical_gt, count_sampson_estimated, count_symmetrical_estimated


def evaluate_all_matching_stats_even_normalized(stats_map_all: dict, tex_save_path_prefix=None, n_examples=None, special_diff=None, scene_info: SceneInfo=None):
    evaluate_all_matching_stats(stats_map_all, tex_save_path_prefix, n_examples, special_diff)


def evaluate_all_matching_stats(stats_map_all: dict, tex_save_path_prefix=None, n_examples=None, special_diff=None, scene_info: SceneInfo=None):
    print("Stats for all difficulties:")

    parameters_keys_list = list(stats_map_all.keys())

    all_diffs = set()
    for key in parameters_keys_list:
        all_diffs = all_diffs.union(set(stats_map_all[key].keys()))
    all_diffs = list(all_diffs)
    all_diffs.sort()

    if n_examples is not None and n_examples > 0:
        for diff in all_diffs:
            if special_diff is not None and special_diff != diff:
                continue
            for key in parameters_keys_list:
                if stats_map_all[key].__contains__(diff): # and len(stats_map_all[key][diff]) > 0:
                    print_significant_instances(stats_map_all[key][diff], key, diff, n_examples=n_examples)


    angle_thresholds = [5, 10, 20]
    accuracy_diff_acc_data_lists = [None] * len(angle_thresholds)
    for angle_threshold in angle_thresholds:

        diff_acc_data_lists = [[] for _ in parameters_keys_list]
        accuracy_diff_acc_data_lists.append(diff_acc_data_lists)

        print("Accuracy({}º) {}".format(angle_threshold, " ".join([str(k) for k in parameters_keys_list])))
        for diff in all_diffs:
            value_list = []
            for i_param_list, key in enumerate(parameters_keys_list):
                if stats_map_all[key].__contains__(diff) and len(stats_map_all[key][diff]) > 0:
                    all_len_const = len(scene_info.img_pairs_lists[diff]) if scene_info is not None else None
                    difficulty, perc = evaluate_percentage_correct(stats_map_all[key][diff], diff, th_degrees=angle_threshold, all_len_const=all_len_const)
                    value_list.append("{:.3f}".format(perc))
                    diff_acc_data_lists[i_param_list].append((float(diff), perc))
                else:
                    value_list.append("--")
            print("{} {}".format(diff, " ".join(value_list)))

    print("Counts {}".format(" ".join([str(k) for k in parameters_keys_list])))
    for diff in all_diffs:
        value_list = []
        for key in parameters_keys_list:
            if scene_info is not None:
                value_list.append(str(len(scene_info.img_pairs_lists[diff])))
            else:
                if stats_map_all[key].__contains__(diff):
                    value_list.append(str(len(stats_map_all[key][diff])))
                else:
                    value_list.append("0")
        print("{} {}".format(diff, " ".join(value_list)))

    if scene_info is not None:
        print("Failed pairs")
        failed_pairs = {}
        for key in parameters_keys_list:
            failed_pairs[key] = set()
        for diff in all_diffs:
            for key in parameters_keys_list:
                if stats_map_all[key].__contains__(diff):
                    relevant_set = set(stats_map_all[key][diff])
                else:
                    relevant_set = set()
                all_from_diff = set([SceneInfo.get_key_from_pair(p) for p in scene_info.img_pairs_lists[diff]])
                failed_pairs[key] = failed_pairs[key].union((all_from_diff - relevant_set))


def print_significant_instances(stats_map, difficulty, key, n_examples=10):

    sorted_by_err_R = list(sorted(stats_map.items(), key=lambda key_value: -key_value[1].error_R))
    print("{} worst examples for {} for diff={}".format(n_examples, key, difficulty))
    for k, v in sorted_by_err_R[:n_examples]:
        print("{}: {}".format(k, v.error_R))
    print("{} best examples for {} for diff={}".format(n_examples, key, difficulty))
    for k, v in sorted_by_err_R[-n_examples:]:
        print("{}: {}".format(k, v.error_R))


def evaluate_percentage_correct(stats_map, difficulty, th_degrees=5, all_len_const=None):

    rad_th = th_degrees * math.pi / 180
    filtered = list(filter(lambda key_value: key_value[1].error_R < rad_th, stats_map.items()))
    filtered_len = len(filtered)
    all_len = all_len_const if all_len_const is not None else len(stats_map.items())
    if all_len == 0:
        return difficulty, 0.0
    else:
        perc = filtered_len/all_len
        return difficulty, perc


def evaluate_stats(stats_map, all):
    evaluate_normals_stats(stats_map)
    evaluate_matching_stats(stats_map)
    if all:
        evaluate_per_img_stats(stats_map)


def evaluate_per_img_stats(stats_map):

    if not stats_map.__contains__("per_img_stats"):
        print("WARNING: 'per_img_stats' not found, skipping")
    else:
        print("Per img stats:")
        m = stats_map["per_img_stats"]
        for configuration in m:
            print("configuration: {}".format(configuration))
            sum_area = 0
            sum_warps = 0
            sum_components = 0
            for img in m[configuration]:
                areas = m[configuration][img].get("affnet_warped_img_size", [])
                affnet_warps_per_component = m[configuration][img].get("affnet_warps_per_component", [])
                sum_components += len(affnet_warps_per_component)
                for warps in affnet_warps_per_component:
                    sum_warps += warps
                for area in areas:
                    sum_area += area
            avg_area = sum_area / len(m[configuration])
            print("avg rectified warped imgs area: {}".format(avg_area))
            avg_warps_per_component = sum_warps / sum_components
            print("avg number of warps per component: {}".format(avg_warps_per_component))


def get_all_diffs(maps_all_params):
    keys_list = maps_all_params.keys()
    all_diffs = set()
    for key in keys_list:
        all_diffs = all_diffs.union(set(maps_all_params[key].keys()))
    all_diffs = list(all_diffs)
    all_diffs.sort()
    return all_diffs


def evaluate_matching_stats(stats_map):

    if not stats_map.__contains__("matching"):
        print("WARNING: 'matching key' not found, skipping")
        return
    matching_map_all = stats_map["matching"]

    stats_local = {"all_keypoints": {}, "tentatives": {}, "inliers": {}, "inlier_ratio": {}}

    all_diffs = get_all_diffs(matching_map_all)
    keys_list = matching_map_all.keys()

    for difficulty in all_diffs:

        stats_local["all_keypoints"][difficulty] = []
        stats_local["tentatives"][difficulty] = []
        stats_local["inliers"][difficulty] = []
        stats_local["inlier_ratio"][difficulty] = []

        for param_key in keys_list:
            matching_map_per_key = matching_map_all[param_key]

            if matching_map_per_key.__contains__(difficulty):
                matching_map = matching_map_per_key[difficulty]

                kps = 0
                tentatives = 0
                inliers = 0
                for pair_name in matching_map:
                    kps = kps + matching_map[pair_name]["kps1"]
                    kps = kps + matching_map[pair_name]["kps2"]
                    tentatives = tentatives + matching_map[pair_name]["tentatives"]
                    inliers = inliers + matching_map[pair_name]["inliers"]

                all_value = kps / (2 * len(matching_map))
                tentatives_value = tentatives / len(matching_map)
                inliers_value = inliers / len(matching_map)
                stats_local["all_keypoints"][difficulty].append("{:.3f}".format(all_value))
                stats_local["tentatives"][difficulty].append("{:.3f}".format(tentatives_value))
                stats_local["inliers"][difficulty].append("{:.3f}".format(inliers_value))
                stats_local["inlier_ratio"][difficulty].append("{:.3f}".format(inliers_value / tentatives_value))
            else:
                stats_local["all_keypoints"][difficulty].append("--")
                stats_local["tentatives"][difficulty].append("--")
                stats_local["inliers"][difficulty].append("--")
                stats_local["inlier_ratio"][difficulty].append("--")


    for key in ["all_keypoints", "tentatives", "inliers", "inlier_ratio"]:
        print("{} across difficulties: ".format(key))
        print("\t".join(keys_list))
        for difficulty in all_diffs:
            print("{}".format("\t".join(stats_local[key][difficulty])))


def evaluate_normals_stats(stats_map):

    normals_degrees = stats_map.get('normals_degrees', None)
    valid_normals = stats_map.get('valid_normals', None)
    if normals_degrees is None or valid_normals is None:
        print("normals will not be evaluated, probably all has been cached")
        return

    def shared_pairs(keys):

        at_least_two_sets = {}
        for k in keys:
            at_least_two_sets[k] = set()
            for img in normals_degrees[k]:
                deg_list = normals_degrees[k][img]
                if len(deg_list) > 0:
                    at_least_two_sets[k].add(img)

        first_key = list(at_least_two_sets.keys())[0]
        shared_keys = at_least_two_sets[first_key]
        for k in at_least_two_sets:
            shared_keys = shared_keys.intersection(at_least_two_sets[k])

        return shared_keys, at_least_two_sets

    shared_at_least_two, _ = shared_pairs(normals_degrees.keys())
    print("{} imgs are common to all keys".format(len(shared_at_least_two)))

    for param_key in normals_degrees:
        count = 0
        count_shared = 0
        count_valid = 0
        avg_l1 = 0.0
        avg_l1_shared = 0.0
        avg_l1_valid = 0.0
        for img in normals_degrees[param_key]:
            deg_list = normals_degrees[param_key][img]
            if valid_normals[param_key][img] > 1:
                count_valid = count_valid + 1
                avg_l1_valid = avg_l1_valid + math.fabs(90.0 - deg_list[0])
            if len(deg_list) > 0:
                avg_l1 = avg_l1 + math.fabs(90.0 - deg_list[0])
                count = count + 1
                if shared_at_least_two.__contains__(img):
                    avg_l1_shared = avg_l1_shared + math.fabs(90.0 - deg_list[0])
                    count_shared = count_shared + 1

        if count_valid > 0:
            avg_l1_valid = avg_l1_valid / count_valid
        if count > 0:
            avg_l1 = avg_l1 / count
        if count_shared > 0:
            avg_l1_shared = avg_l1_shared / count_shared
        print("{} {:.3f} {} / {}".format(param_key, avg_l1, count, count_valid))
        print("{} - shared: {:.3f}/{} valid: {:.3f}/{}".format(param_key, avg_l1_shared, count_shared, avg_l1_valid, count_valid))
