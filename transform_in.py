import paddle

from clicker import Click
import paddle.nn.functional as F


class BaseTransform(object):
    def __init__(self):
        self.image_changed = False

    def transform(self, image_nd, clicks_lists):
        raise NotImplementedError

    def inv_transform(self, prob_map):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class SigmoidForPred(BaseTransform):
    def transform(self, image_nd, clicks_lists):
        return image_nd, clicks_lists

    def inv_transform(self, prob_map):
        return F.sigmoid(prob_map)

    def reset(self):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass



class AddHorizontalFlip3D(BaseTransform):
    def transform(self, image_nd, clicks_lists):
        assert len(image_nd.shape) == 5, "len(image_nd.shape) should be 5, but it equals to {}".format(len(image_nd.shape))
        image_nd = paddle.concat([image_nd, paddle.flip(image_nd, axis=[4])], axis=0)

        image_depth = image_nd.shape[4]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [click.copy(coords=(click.coords[0], click.coords[1], image_depth - click.coords[2] - 1))
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        return image_nd, clicks_lists

    def inv_transform(self, prob_map):
        assert len(prob_map.shape) == 5 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]

        return 0.5 * (prob_map + paddle.flip(prob_map_flipped, axis=[3]))

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def reset(self):
        pass
