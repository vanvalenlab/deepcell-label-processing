import numpy as np
import pickle
from skimage import exposure
import yaml

import constants


def load_embeddings(prediction_file):
    """ Load relevant variables from output of deepcell-types """
    with open(prediction_file, 'rb') as f:
        cell_indices, preds_embedding, preds_celltype_probs, y_pred_celltype, y_true_celltype = pickle.load(
            f)
    return cell_indices, preds_embedding, preds_celltype_probs, y_pred_celltype, y_true_celltype


def load_raw(raw_file, ground_truth=False):
    """ Load raw data from a raw file """
    data = np.load(raw_file, allow_pickle=True)
    X = data['X']
    y = data['y']
    if ground_truth:
        cell_types = data['cell_types'].item()
        return X, y, cell_types
    return X, y


def parse_kept_channels(config_file):
    """ Obtain the relevant channels from a dataset's config file """
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        return config['channels_to_keep']


def parse_metadata(metadata_file, kept_channels):
    """ Obtain relevant channel indices and cell type mapper from metadata file """
    with open(metadata_file, 'r') as stream:
        channels = []
        channel_indices = []
        metadata = yaml.safe_load(stream)
        try:
            mapper = metadata['meta']['file_contents']['cell_types']['mapper']
        except:
            mapper = None
        for channel in metadata['meta']['sample']['channels']:
            if channel['target'] in kept_channels:
                channels.append(kept_channels[channel['target']])
                channel_indices.append(channel['index'])
    return channel_indices, channels, mapper


def get_all_channels(metadata_file):
    """ Obtain relevant channel indices and cell type mapper from metadata file """
    with open(metadata_file, 'r') as stream:
        channels = []
        channel_indices = []
        metadata = yaml.safe_load(stream)
        try:
            mapper = metadata['meta']['file_contents']['cell_types']['mapper']
        except:
            mapper = None
        for channel in metadata['meta']['sample']['channels']:
            channels.append(channel['target'])
            channel_indices.append(channel['index'])
    return channel_indices, channels, mapper


def parse_groundtruth(cell_types, mapper):
    """ Construct cellTypes.json from ground truth cell types mapping """
    cell_types_json = []
    counter = 0
    for cell_type_id in mapper:
        cells = []
        for k, v in cell_types.items():
            if cell_types[k] == cell_type_id:
                cells.append(k)
        cell_types_json.append({'id': cell_type_id, 'cells': cells,
                               'color': constants.COLOR_MAP[counter], 'name': mapper[cell_type_id], 'feature': 0})
        counter += 1
    cell_types_json = cell_types_json[1:]  # remove background
    return cell_types_json


def parse_predictions(y_pred_celltype, cell_indices):
    """ Construct cellTypes.json from deepcell-types output """
    cell_types_json = []
    for i in range(1, len(constants.MASTER_TYPES) + 1):
        cells = []
        for j in range(len(y_pred_celltype)):
            celltype_id = y_pred_celltype[j]
            if celltype_id == i:
                cells.append(int(cell_indices[j]))
        cell_types_json.append(
            {'id': i, 'cells': cells, 'color': constants.COLOR_MAP[i - 1], 'name': constants.MASTER_TYPES[i - 1], 'feature': 0})
    return cell_types_json


def make_empty_cell_types():
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i in range(1, len(constants.MASTER_TYPES) + 1):
        cell_types_json.append({'id': i, 'cells': [], 'color': constants.COLOR_MAP[i - 1],
                               'name': constants.MASTER_TYPES[i - 1], 'feature': 0})
    return cell_types_json


def make_empty_marker_positivity(channels):
    """ Construct cellTypes.json for a marker positivity job """
    cell_types_json = []
    for i in range(len(channels)):
        cell_types_json.append({'id': i + 1, 'cells': [], 'color': constants.COLOR_MAP[(i - 1) % len(constants.COLOR_MAP)],
                                'name': channels[i], 'feature': 0})
    return cell_types_json


def parse_embeddings(preds_embedding, cell_indices):
    """ Construct embeddings.json array from deepcell-types output """
    embeddings = np.zeros((np.max(cell_indices) + 1, preds_embedding[0].size))
    for i in range(len(cell_indices)):
        embeddings[cell_indices[i]] = preds_embedding[i]
    return embeddings


def reshape_X(X, channel_indices):
    """ Reshape X assuming (TYXC) order => (CTYX) and remove irrelevant channels """
    return np.take(X.transpose(3, 0, 1, 2), channel_indices, 0)


def reshape_y(y):
    """ Reshape y assuming (TYXC) => (CTYX) """
    return y.transpose(3, 0, 1, 2)


def tile_around_center(array, num_tiles, size_x, size_y):
    """ Given an array with dimension order (C,T,Y,X), crop a rectangle of size (num_tiles * size_x) by (num_tiles * size_y) around the center of the X,Y array,
        then crop that square into (num_tiles * num_tiles) tiles of size (size_x, size_y), and finally stack the arrays along the T axis and return """
    center_x = int(array.shape[3] / 2)
    center_y = int(array.shape[2] / 2)
    crop = array[:, :, center_y - int(num_tiles * size_y / 2):center_y + int(num_tiles * size_y / 2),
                 center_x - int(num_tiles * size_x / 2):center_x + int(num_tiles * size_x / 2)]
    batches = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            batches.append(crop[:, 0, i * size_y: i * size_y +
                                size_y, j * size_x:j*size_x + size_x])
    return np.moveaxis(np.stack(batches), 0, 1)


def tile_and_stack_array(array, size_x, size_y):
    """ Try to exactly crop and stack an array into tiles of size (size_x, size_y) if possible """
    try:
        if array.shape[2] % size_y != 0:
            raise Exception(
                f"Array of shape {array.shape} cannot be divided equally into tiles of size {size_x} by {size_y}")
        elif array.shape[3] % size_x != 0:
            raise Exception(
                f"Array of shape {array.shape} cannot be divided equally into tiles of size {size_x} by {size_y}")
        else:
            batches = []
            for i in range(int(array.shape[3] / size_x)):
                for j in range(int(array.shape[2] / size_y)):
                    crop = array[:, 0, i * size_y: i * size_y +
                                 size_y, j * size_x:j*size_x + size_x]
                    batches.append(crop)
            tiled = np.moveaxis(np.stack(batches), 0, 1)
            return tiled
    except Exception as e:
        print(e)


def to_int32(y):
    """ Change dtype to int32 """
    return y.astype('int32')


def normalize_raw(X):
    """ Rescale raw image to 0-255 uint8 values """
    channel_maxes = np.max(X, axis=(1, 2, 3), keepdims=True)
    channel_mins = np.min(X, axis=(1, 2, 3), keepdims=True)
    norm_X = (X - channel_mins) / (channel_maxes - channel_mins) * 255
    return norm_X.astype('uint8')


def equalize_adapthist(X):
    """ Run the adaptive histogram equalization algorithm on the raw image """
    return exposure.equalize_adapthist(X, clip_limit=0.1)
