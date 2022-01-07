import pydegensac

from config import Config
from scene_info import *
from utils import *


def decolorize(img):
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2):
    h = img1.shape[0]
    w = img1.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    def possibly_decolorize(img_local):
        if len(img_local.shape) <= 2:
            return img2
        return decolorize(img_local)

    img1_dec = possibly_decolorize(img1)
    img2_dec = possibly_decolorize(img2)

    if H is not None:
        dst = cv.perspectiveTransform(pts, H)
        img2_tr = cv.polylines(img2_dec, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
    else:
        img2_tr = img2_dec

    matches_mask = inlier_mask.ravel().tolist()

    # Blue is estimated homography
    draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=20)
    img_out = cv.drawMatches(img1_dec, kps1, img2_tr, kps2, tentative_matches, None, **draw_params)
    return img_out


def rich_split_points(tentative_matches, kps1, dsc1, kps2, dsc2):

    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    src_kps = [kps1[m.queryIdx] for m in tentative_matches]
    src_dsc = [dsc1[m.queryIdx] for m in tentative_matches]

    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_kps = [kps2[m.trainIdx] for m in tentative_matches]
    dst_dsc = [dsc2[m.trainIdx] for m in tentative_matches]

    return src_pts, src_kps, src_dsc, dst_pts, dst_kps, dst_dsc


def get_cross_checked_tentatives(matcher, img_data1, img_data2, ratio_threshold):

    knn_matches = matcher.knnMatch(img_data1.descriptions, img_data2.descriptions, k=2)
    matches2 = matcher.match(img_data2.descriptions, img_data1.descriptions)

    tentative_matches = []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue
        if m.distance < ratio_threshold * n.distance:
            tentative_matches.append(m)

    filter_on_planes_during_correspondence = False
    if is_rectified_condition(img_data1) and filter_on_planes_during_correspondence:
        tentative_matches = filter_during_correspondence(matcher, tentative_matches, matches2, img_data1, img_data2, ratio_threshold)
    return tentative_matches


def filter_during_correspondence(matcher, tentative_matches, all_matches_reversed, img_data1, img_data2, ratio_threshold):

    src_pts, dst_pts = split_points(tentative_matches, img_data1.key_points, img_data2.key_points)

    stats, unique, counts = get_normals_stats([img_data1, img_data2], src_pts, dst_pts)

    max_set = get_filter(stats, unique, counts, img_data1.normals.shape[0], img_data2.normals.shape[0])

    all_src_pts = np.float32([img_data1.key_points[m.trainIdx].pt for m in all_matches_reversed]).reshape(-1, 2)
    all_dst_pts = np.float32([img_data2.key_points[m.queryIdx].pt for m in all_matches_reversed]).reshape(-1, 2)
    all_matches_stats, all_uniques, all_counts = get_normals_stats([img_data1, img_data2], all_src_pts, all_dst_pts)

    new_cross_adds = 0
    new_adds = 0
    unchecked_adds = 0
    tentative_matches = []

    knn_matches = matcher.knnMatch(img_data1.descriptions, img_data2.descriptions, k=3)

    for knn_match in knn_matches:
        for match_idx, match in enumerate(knn_match):
            key = tuple(all_matches_stats[match.trainIdx, :])
            if key[0] != -1 and key[1] != -1 and not max_set.__contains__(key):
                continue
            if all_matches_reversed[match.trainIdx].trainIdx != match.queryIdx:
                break
            elif match_idx > 0:
                new_cross_adds += 1
            if match_idx < 2:
                if match.distance < ratio_threshold * knn_match[match_idx + 1].distance:  # ratio_threshold was 0.85
                    tentative_matches.append(match)
                    if match_idx > 0:
                        new_adds += 1
                    break
            else:
                unchecked_adds += 1
                tentative_matches.append(match)
                break

    return tentative_matches


def filter_fginn_matches(matches, desc1, desc2, num_nn, cfg):
    """
    :param matches:
    :param desc1:
    :param desc2:
    :param num_nn:
    :param cfg:
    :return:

    FGINN — 1st geometrically inconsistent nearest neighbor ratio
    https://ducha-aiki.medium.com/how-to-match-to-learn-or-not-to-learn-part-2-1ab52ede2022
    """

    valid_matches = []

    fginn_spatial_th = cfg["fginn_spatial_th"]
    ratio_th = cfg["ratio_th"]

    if fginn_spatial_th < 0 or fginn_spatial_th > 500:
        raise ValueError('FGINN radius outside expected range')

    if ratio_th < 0.1 or ratio_th > 1.01:
        raise ValueError('Ratio test threshold outside expected range')

    flann = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    not_fginns = flann.radiusMatch(desc2.astype(np.float32),
                                   desc2.astype(np.float32),
                                   fginn_spatial_th)

    for m_idx, cur_match in enumerate(matches):
        for mii in range(num_nn):
            cur_non_fginns = [
                x.trainIdx for x in not_fginns[cur_match[mii].trainIdx]
            ]
            for nn_2 in cur_match[mii + 1:]:
                if cur_match[mii].distance <= ratio_th * nn_2.distance:
                    valid_matches.append(cur_match[mii])
                    break
                if nn_2.trainIdx not in cur_non_fginns:
                    break

    return valid_matches


def find_correspondences(img_data1,
                         img_data2,
                         cfg, out_dir=None, save_suffix=None, ratio_thresh=None, show=True, save=True):

    fginn = cfg["fginn"]
    num_nn = cfg["num_nn"]

    crossCheck = False
    if Config.do_flann():
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=128, crossCheck=crossCheck)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv.BFMatcher(crossCheck=crossCheck)

    assert img_data1.descriptions is not None and len(img_data1.descriptions) != 0
    assert img_data2.descriptions is not None and len(img_data2.descriptions) != 0

    if fginn:
        k = 10 + num_nn
        knn_matches = matcher.knnMatch(img_data1.descriptions, img_data2.descriptions, k=k)
        tentative_matches = filter_fginn_matches(knn_matches, img_data1.descriptions, img_data2.descriptions, num_nn, cfg)
    else:
        tentative_matches = get_cross_checked_tentatives(matcher, img_data1, img_data2, ratio_thresh)

    if show or save:
        tentative_matches_in_singleton_list = [[m] for m in tentative_matches]
        img3 = cv.drawMatchesKnn(img_data1.img, img_data1.key_points, img_data2.img, img_data2.key_points, tentative_matches_in_singleton_list, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("auto")
        plt.title("{} tentative matches".format(len(tentative_matches)))
        plt.imshow(img3)
        if save:
            assert out_dir is not None
            assert save_suffix is not None
            plt.savefig("{}/tentative_{}.jpg".format(out_dir, save_suffix))
        show_or_close(show)

    return tentative_matches


def find_keypoints(scene_name, image_entry: ImageEntry, descriptor):

    img_path = 'original_dataset/{}/images/{}.jpg'.format(scene_name, image_entry.image_name)
    img = cv.imread(img_path)
    if img is None:
        return None, None
    kps, descs = descriptor.detectAndCompute(img, None)
    return kps, descs


def show_save_matching(img1,
                       kps1,
                       img2,
                       kps2,
                       tentative_matches,
                       inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save):

    if show or save:
        img_matches = draw_matches(kps1, kps2, tentative_matches, None, inlier_mask, img1, img2)
        plt.figure()
        inliers_count = np.sum(inlier_mask)
        plt.title("Matches in line with the model - {} inliers".format(inliers_count))
        plt.imshow(img_matches)
        if save:
            plt.savefig("{}/matches_{}.jpg".format(out_dir, save_suffix))
        show_or_close(show)


def match_epipolar(img_data1,
                   img_data2,
                   find_fundamental, img_pair, out_dir, show, save, ratio_thresh,
                   ransac_th, ransac_conf, ransac_iters, cfg):

    Timer.start_check_point("matching")

    save_suffix = "{}_{}".format(img_pair.img1, img_pair.img2)

    tentative_matches = find_correspondences(img_data1,
                                             img_data2,
                                             cfg, out_dir, save_suffix, ratio_thresh=ratio_thresh, show=show, save=save)

    src_pts, dst_pts = split_points(tentative_matches, img_data1.key_points, img_data2.key_points)

    if find_fundamental:
        F, inlier_mask = cv.findFundamentalMat(src_pts, dst_pts, method=cv.USAC_MAGSAC, ransacReprojThreshold=ransac_th, confidence=ransac_conf, maxIters=ransac_iters)
        if F is None or inlier_mask is None:
            print("WARNING: F:{} or inlier mask:{} are None".format(F, inlier_mask))
            raise ValueError("None")
        E = img_data2.real_K.T @ F @ img_data1.real_K
    else:
        E, inlier_mask = cv.findEssentialMat(src_pts, dst_pts, img_data1.real_K, None, img_data2.real_K, None, cv.RANSAC, prob=ransac_conf, threshold=ransac_th)
        if E is None or inlier_mask is None:
            print("WARNING: E:{} or inlier mask:{} are None".format(E, inlier_mask))
            raise ValueError("None")

    Timer.end_check_point("matching")

    show_save_matching(img_data1.img,
                       img_data1.key_points,
                       img_data2.img,
                       img_data2.key_points,
                       tentative_matches,
                       inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save)

    return E, inlier_mask, src_pts, dst_pts, tentative_matches


def match_find_F_degensac(img1, kps1, descs1, real_K_1, img2, kps2, descs2, real_K_2, img_pair, out_dir, show, save, ratio_thresh, ransac_th, ransac_conf, ransac_iters):

    Timer.start_check_point("matching")

    save_suffix = "{}_{}".format(img_pair.img1, img_pair.img2)

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, out_dir, save_suffix, ratio_thresh=ratio_thresh, show=show, save=save)

    src_pts, dst_pts = split_points(tentative_matches, kps1, kps2)

    F, inlier_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, px_th=ransac_th, conf=ransac_conf, max_iters=ransac_iters, enable_degeneracy_check=True)
    inlier_mask = np.expand_dims(inlier_mask, axis=1)

    E = real_K_2.T @ F @ real_K_1

    Timer.end_check_point("matching")

    show_save_matching(img1,
                       kps1,
                       img2,
                       kps2,
                       tentative_matches,
                       inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save)

    return E, inlier_mask, src_pts, dst_pts, tentative_matches
