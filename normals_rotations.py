import scipy.optimize

from evaluation import *
from utils import get_rotation_matrix_safe


@dataclass
class RotationSolution:

    objective_fnc: float
    rotation_vector_normal: float
    rotation_vector: np.ndarray
    permutation: list


def get_distance(one, two, angular_distance=True, squared=True):

    if angular_distance:
        scalar_products = one[:, 0] * two[:, 0] + one[:, 1] * two[:, 1] + one[:, 2] * two[:, 2]
        scalar_products = np.clip(scalar_products, a_min=-1.0, a_max=1.0)
        angular_distances = np.arccos(scalar_products)
        if squared:
            dist = np.sum(angular_distances ** 2) / angular_distances.shape[0]
        else:
            dist = np.sum(np.abs(angular_distances)) / angular_distances.shape[0]
    else:
        norms = np.linalg.norm(one - two, axis=1)
        if squared:
            dist = np.sum(norms ** 2) / norms.shape[0]
        else:
            dist = np.sum(np.abs(norms)) / norms.shape[0]

    return dist


def rotation_sort_value(solution: RotationSolution):
    return math.sqrt(solution.objective_fnc) * 10 + solution.rotation_vector_normal


def find_sorted_rotations(normals1, normals2, zero_around_z):

    assert len(normals1.shape) == 2
    assert len(normals2.shape) == 2

    assert np.all(np.abs(np.linalg.norm(normals1, axis=1) - 1.0) < 10 ** -5)
    assert np.all(np.abs(np.linalg.norm(normals2, axis=1) - 1.0) < 10 ** -5)

    solutions = find_rotations_info(normals1, normals2, zero_around_z)
    solutions.sort(key=lambda solution: rotation_sort_value(solution))
    return solutions


class ObjFunction:

    def __init__(self, src_normals, dst_normals, zero_around_z):
        self.src_normals = src_normals
        self.dst_normals = dst_normals
        self.zero_around_z = zero_around_z

    def function_value(self, r):

        if self.zero_around_z:
            r = np.array([r[0], r[1], 0.0])

        R = get_rotation_matrix_safe(r)
        rotated = (R @ self.src_normals.T).T

        return get_distance(rotated, self.dst_normals)


def find_rotations_info(normals1, normals2, zero_around_z):

    swapped = False
    if len(normals1) > len(normals2):
        swapped = True
        normals1, normals2 = normals2, normals1

    len1 = len(normals1)
    len2 = len(normals2)

    solutions = []

    for cur_permutation in itertools.permutations(range(len2), len1):

        def min_function(r):
            target = normals2[list(cur_permutation)]
            return ObjFunction(normals1, target, zero_around_z).function_value(r)

        use_shgo = False
        if use_shgo:
            sf = 1.0
            if zero_around_z:
                bounds = [(-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf)]
            else:
                bounds = [(-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf)]
            solution = scipy.optimize.shgo(min_function, bounds)
            if solution.success:

                for i in range(solution.xl.shape[0]):
                    r_star = solution.xl[i]
                    value = min_function(r_star)

                if swapped:
                    r_star = -r_star
                if zero_around_z:
                    r_star = np.array([r_star[0], r_star[1], 0])
                solutions.append(RotationSolution(rotation_vector=r_star, objective_fnc=value, rotation_vector_normal=np.linalg.norm(r_star), permutation=list(cur_permutation)))
        else:
            if zero_around_z:
                initial = np.zeros(2)
            else:
                initial = np.zeros(3)
            r_star = scipy.optimize.fmin(min_function, initial, disp=False)
            value = min_function(r_star)
            if swapped:
                r_star = -r_star
            if zero_around_z:
                r_star = np.array([r_star[0], r_star[1], 0])
            solutions.append(RotationSolution(rotation_vector=r_star, objective_fnc=value, rotation_vector_normal=np.linalg.norm(r_star), permutation=list(cur_permutation)))

    return solutions
