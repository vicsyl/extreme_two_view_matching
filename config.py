import cv2 as cv


class Config:

    key_do_flann = "do_flann"
    key_planes_based_matching_merge_components = "key_planes_based_matching_merge_components"

    # Toft et al. use 80 (but the implementation details actually differ)
    plane_threshold_degrees = 75

    rectification_interpolation_key = "rectification_interpolation_key"

    # window size

    # init the map and set the default values
    config_map = {}
    config_map[key_do_flann] = False
    config_map[key_planes_based_matching_merge_components] = True
    config_map[rectification_interpolation_key] = cv.INTER_LINEAR

    @staticmethod
    def log():
        print("Config:")
        print("\t{}".format("\n\t".join("{}\t{}".format(k, v) for k, v in Config.config_map.items())))

        attr_list = [attr for attr in dir(Config) if not callable(getattr(Config, attr)) and not attr.startswith("__")]
        for attr_name in attr_list:
            print("\t{}\t{}".format(attr_name, getattr(Config, attr_name)))
        print()

    @staticmethod
    def do_flann():
        return Config.config_map[Config.key_do_flann]


class Property:

    # NOTES cache_img_data is most likely correct for majority of props
    cartesian_values = "cartesian_values"
    cache_normals = 0
    cache_clusters = 1
    cache_img_data = 2
    all_combinations = 3

    def __init__(self, _type, default, optional=False, cache=cache_img_data, list_allowed=True, allowed_values=None):
        self.type = _type
        self.default = default
        self.optional = optional
        self.cache = cache
        self.list_allowed = list_allowed
        self.allowed_values = allowed_values

    def parse_and_update(self, key, value, config):

        assert key != Property.cartesian_values

        if Property.is_list(value):
            if not self.list_allowed:
                raise ValueError("List allowed. Value: {}".format(value))
            raw_list = Property.parse_list(value[1:-1])
            if not config.__contains__(Property.cartesian_values):
                config[Property.cartesian_values] = {}
            config[Property.cartesian_values][key] = [self.parse_value(i) for i in raw_list]
        else:
            config[key] = self.parse_value(value)

    def parse_value(self, value):

        if self.optional and value.lower() == "none":
            return None

        if self.type == "string":
            return value
        elif self.type == "bool":
            return value.lower() == "true"
        elif self.type == "float":
            return float(value)
        elif self.type == "int":
            return int(value)
        elif self.type == "list":
            return Property.parse_list(value)
        elif self.type == "enum":
            if value not in self.allowed_values:
                raise ValueError("value '{}' not allowed - expected one on [{}]".format(value, ", ".join([str(i) for i in self.allowed_values])))
            return value
        else:
            raise ValueError("unknown type: {}".format(self.type))


    @staticmethod
    def parse_list(list_str: str):
        fields = list_str.split(",")
        fields = filter(lambda x: x != "", map(lambda x: x.strip(), fields))
        fields = list(fields)
        return fields

    @staticmethod
    def is_list(v):
        stripped = v.strip()
        return stripped.startswith("[") and stripped.endswith("]")


class CartesianConfig:

    config_combination = "config_combination"
    max_one_non_default = "max_one_non_default"
    just_one_non_default = "just_one_non_default"
    cartesian = "cartesian"

    angle_distance_threshold_degrees = "angle_distance_threshold_degrees"
    filter_sky = "filter_sky"
    all_unrectified = "all_unrectified"
    rectify = "rectify"

    props_handlers = {

        # dataset
        "scene_name": Property("string", default=False, cache=Property.cache_img_data),
        "scene_type": Property("enum", default="orig", cache=Property.cache_img_data, allowed_values=["orig", "google"]),

        # preprocessing
        "handle_antipodal_points": Property("bool", default=False, cache=Property.cache_img_data),
        "svd_weighted":  Property("bool", default=True, cache=Property.cache_img_data),
        "mean_shift_type": Property("enum", optional=True, default="mean", cache=Property.cache_img_data, allowed_values=["mean", "full"]),
        "singular_value_quantil": Property("float", default=1.0, cache=Property.cache_img_data),
        angle_distance_threshold_degrees: Property("int", default=35, cache=Property.cache_img_data),
        filter_sky: Property("bool", default=True, cache=Property.cache_img_data),
        all_unrectified: Property("bool", default=False, cache=Property.cache_img_data),
        rectify: Property("bool", default=True, cache=Property.cache_img_data),
        "svd_weighted_sigma": Property("float", default=0.8, cache=Property.cache_img_data),

        # fginn
        "fginn": Property("bool", False, cache=Property.cache_img_data),
        "num_nn": Property("int", 2, cache=Property.cache_img_data),
        "fginn_spatial_th": Property("int", 100, cache=Property.cache_img_data),
        "ratio_th": Property("float", 0.5, cache=Property.cache_img_data),
        "feature_descriptor": Property("enum", default="SIFT", cache=Property.cache_img_data, allowed_values=["SIFT", "BRISK", "SUPERPOINT", "ROOT_SIFT", "HARD_NET"]),

        "pipeline_final_step": Property("enum", default="final", cache=Property.all_combinations, list_allowed=False, allowed_values=["final", "before_matching", "before_rectification"]),
        "rectify_by_fixed_rotation": Property("bool", default=False, cache=Property.all_combinations),
        "rectify_by_0_around_z": Property("bool", default=False, cache=Property.all_combinations),
        "rectify_by_GT": Property("bool", default=False, cache=Property.all_combinations),
        "rotation_alpha1": Property("float", default=1.0, cache=Property.cache_img_data, list_allowed=True),
        "rotation_alpha2": Property("float", default=1.0, cache=Property.cache_img_data, list_allowed=True),
        "rectify_affine_affnet": Property("bool", default=False, cache=Property.cache_img_data, list_allowed=True),

        # AFFNET
        # param for naive approach via "mean"
        "affnet_tilt_r_ln": Property("float", default=1.7, cache=Property.cache_img_data, list_allowed=True),
        # param for naive approach via "mean"
        "affnet_max_tilt_r": Property("float", default=5.8, cache=Property.cache_img_data, list_allowed=True),
        "affnet_hard_net_filter": Property("int", default=None, optional=True, cache=Property.cache_img_data, list_allowed=False),
        "show_affnet": Property("bool", default=False, optional=True, cache=Property.cache_img_data, list_allowed=False),
        "affnet_include_all_from_identity": Property("bool", default=True, optional=True, cache=Property.cache_img_data),
        "affnet_covering_type": Property("enum", default="dense_cover", cache=Property.cache_img_data, optional=False, list_allowed=True, allowed_values=["dense_cover", "sparse_cover", "dense_cover_original"]),
        "affnet_covering_fraction_th": Property("float", default=0.9, cache=Property.cache_img_data, optional=False, list_allowed=True),
        "affnet_covering_max_iter": Property("int", default=2, cache=Property.cache_img_data, optional=False, list_allowed=True),
        "affnet_no_clustering": Property("bool", default=False, cache=Property.cache_img_data, optional=False, list_allowed=True),

        # SIFT
        "n_features": Property("int", None, optional=True, cache=Property.cache_img_data),
        "sift_octave_layers": Property("int", 3, optional=True, cache=Property.cache_img_data),
        "sift_contrast_threshold": Property("float", 0.04, optional=True, cache=Property.cache_img_data), # try 0.03
        "sift_edge_threshold": Property("int", 10, optional=True, cache=Property.cache_img_data),
        "sift_sigma": Property("float", 1.6, optional=True, cache=Property.cache_img_data),

        # IMG preprocessing
        "img_read_mode": Property("enum", default="RGB", optional=True, list_allowed=False, allowed_values=["RGB", "GRAY"]),
        "img_max_size": Property("int", default=None, optional=True, list_allowed=False),

        #special
        "sequential_files_limit": Property("int", default=None, optional=True, list_allowed=False),
    }

    @staticmethod
    def get_default_cfg():
        ret = {k: v.default for (k, v) in CartesianConfig.props_handlers.items()}
        return ret

    @staticmethod
    def config_parse_line(key: str, value: str, cfg_map):
        if CartesianConfig.props_handlers.__contains__(key):
            CartesianConfig.props_handlers[key].parse_and_update(key, value, cfg_map)
        elif key == CartesianConfig.config_combination:
            if value in [CartesianConfig.max_one_non_default, CartesianConfig.just_one_non_default, CartesianConfig.cartesian]:
                cfg_map[CartesianConfig.config_combination] = value
            else:
                raise ValueError("unexpected value of {} as {}".format(value, CartesianConfig.config_combination))
        else:
            print("WARNING - unrecognized param: {}".format(key))

    @staticmethod
    def print_config(cfg_map):
        for k in cfg_map:
            print("{} = {}".format(k, cfg_map[k]))

    @staticmethod
    def get_default_cache_keys():
        return {
            Property.cache_normals: "",
            Property.cache_clusters: "",
            Property.cache_img_data: "",
            Property.all_combinations: "",
        }

    @staticmethod
    def get_configs(cfg_map, config_combination=None):

        if not cfg_map.__contains__(Property.cartesian_values):
            return [(cfg_map.copy(), CartesianConfig.get_default_cache_keys())]

        if config_combination is None:
            if not cfg_map.__contains__(CartesianConfig.config_combination):
                raise ValueError("combination type unknown")
            config_combination = cfg_map[CartesianConfig.config_combination]

        if config_combination not in [CartesianConfig.max_one_non_default, CartesianConfig.just_one_non_default, CartesianConfig.cartesian]:
            raise ValueError("unexpected value of {} as {}".format(config_combination, CartesianConfig.config_combination))

        comb_list = list(cfg_map.get(Property.cartesian_values, {}).items())
        comb_list.sort(key=lambda k_v: CartesianConfig.props_handlers[k_v[0]].cache)
        comb_list = [(k, v, 0) for (k, v) in comb_list]

        all_conf_cache_keys = []

        def get_new_config():

            cache_keys = CartesianConfig.get_default_cache_keys()

            new_cfg = cfg_map.copy()
            del new_cfg[Property.cartesian_values]
            for key, lst, counter in comb_list:
                value = lst[counter]
                new_cfg[key] = value
                key_v_str = "{}_{}".format(key, value)
                cache = CartesianConfig.props_handlers[key].cache
                for cache_level in range(cache, Property.all_combinations + 1):
                    cache_keys[cache_level] = key_v_str if len(cache_keys[cache_level]) == 0 else "{}_{}".format(cache_keys[cache_level], key_v_str)
            return new_cfg, cache_keys

        done = False
        while not done:

            if config_combination == CartesianConfig.cartesian:
                all_conf_cache_keys.append(get_new_config())
            else:
                # NOTE it will always iterate through angle_distance_threshold_degrees and won't affect the combination type
                non_zeros = [c for (k, l, c) in comb_list if c > 0 and k != CartesianConfig.angle_distance_threshold_degrees]
                if config_combination == CartesianConfig.just_one_non_default and len(non_zeros) == 1:
                    all_conf_cache_keys.append(get_new_config())
                elif config_combination == CartesianConfig.max_one_non_default and len(non_zeros) < 2:
                    all_conf_cache_keys.append(get_new_config())

            index = 0
            (key, lst, counter) = comb_list[index]
            while counter == len(lst) - 1:
                comb_list[index] = (key, lst, 0)
                index = index + 1
                if index == len(comb_list):
                    done = True
                    break
                (key, lst, counter) = comb_list[index]
            if not done:
                comb_list[index] = (key, lst, counter + 1)

        return all_conf_cache_keys

