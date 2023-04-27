import pickle
import numpy as np
import yaml
import io
import zipfile
import json
from tifffile import TiffWriter
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure
from . import constants

def load_embeddings(prediction_file):
    """ Load relevant variables from output of deepcell-types """
    with open(prediction_file, 'rb') as f:
        cell_indices, preds_embedding, preds_celltype_probs, y_pred_celltype, y_true_celltype = pickle.load(f)
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

def parse_metadata(metadata_file, kept_channels):
    """ Obtain relevant channel indices and cell type mapper from metadata file """
    with open(metadata_file, 'r') as stream:
        channels = []
        metadata = yaml.safe_load(stream)
        mapper = metadata['meta']['file_contents']['cell_types']['mapper']
        for channel in metadata['meta']['sample']['channels']:
            channels.append(channel['target'])
    channel_indices = []
    for i in range(len(channels)):
        if channels[i] in kept_channels:
            channel_indices.append(i)
    return channel_indices, mapper

def parse_groundtruth(cell_types, mapper):
    """ Construct cellTypes.json from ground truth cell types mapping """
    cell_types_json = []
    counter = 0
    for cell_type_id in mapper:
        cells = []
        for k, v in cell_types.items():
            if cell_types[k] == cell_type_id:
                cells.append(k)
        cell_types_json.append({'id': cell_type_id, 'cells': cells, 'color': constants.COLOR_MAP[counter], 'name': mapper[cell_type_id], 'feature': 0})
        counter += 1
    cell_types_json = cell_types_json[1:] # remove background
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
        cell_types_json.append({'id': i, 'cells': cells, 'color': constants.COLOR_MAP[i - 1], 'name': constants.MASTER_TYPES[i - 1], 'feature': 0})
    return cell_types_json

def parse_embeddings(preds_embedding, cell_indices):
    """ Construct embeddings.json array from deepcell-types output """
    embeddings = np.zeros((np.max(cell_indices) + 1, preds_embedding[0].size))
    for i in range(len(cell_indices)):
        embeddings[cell_indices[i]] = preds_embedding[i]
    return embeddings

