from tifffile import TiffWriter

import argparse
import io
import json
import os
import requests
import zipfile

import utils


def raw_to_dcl(tile_x, tile_y, tile_center, ground_truth, marker_positivity, clahe, file_path, metadata, config):
    """Converts raw npz file into DCL zip file.

    Args:
        tile_x (int): tile size in x direction
        tile_y (in): tile size in y direction
        tile_center (bool): whether or not to tile around the center
        ground_truth (bool): whether or not to use ground truth
        marker_positivity (bool): whether or not to set up a marker positivity style project
        clahe (bool): whether or not to use CLAHE normalization
        file_path (str): file path to raw npz file
        metadata (str): file path to metadata file
        config (str): file path to config file

    Returns:
        X_processed (np.array): processed raw image
        y_processed (np.array): processed segmentation mask
        cell_types (list): cell types json
        channels (list): list of channels
    """

    print('Loading raw file...\n')
    if ground_truth:
        print('Loading ground truth...\n')
        X, y, cell_types = utils.load_raw(file_path, ground_truth=True)
    else:
        X, y = utils.load_raw(file_path)

    print('Parsing config file...\n')
    kept_channels = utils.parse_kept_channels(config)

    print('Parsing metadata file...\n')
    channel_indices, channels, mapper = utils.parse_metadata(
        metadata, kept_channels)

    print('Making cellTypes.json...\n')
    if marker_positivity:
        cell_types = utils.make_empty_marker_positivity(channels)
    elif ground_truth:
        cell_types = utils.parse_groundtruth(cell_types, mapper)
    else:
        cell_types = utils.make_empty_cell_types()

    print('Making X.ome.tiff...\n')
    if clahe:
        X_processed = utils.equalize_adapthist(
            utils.normalize_raw(utils.reshape_X(X, channel_indices)))
    else:
        X_processed = utils.normalize_raw(utils.reshape_X(X, channel_indices))

    print('Making y.ome.tiff...\n')
    y_processed = utils.to_int32(utils.reshape_y(y))

    if tile_center and tile_x and tile_y:
        print("Tiling X and y around center...")
        X_processed = utils.tile_around_center(X_processed, int(tile_center), int(tile_x), int(tile_y))
        y_processed = utils.tile_around_center(y_processed, int(tile_center), int(tile_x), int(tile_y))
    elif tile_x and tile_y:
        print("Tiling X and y...")
        X_processed = utils.tile_and_stack_array(
            X_processed, int(tile_x), int(tile_y))
        y_processed = utils.tile_and_stack_array(
            y_processed, int(tile_x), int(tile_y))

    return X_processed, y_processed, cell_types, channels


def dcl_zip(X, y, cell_types, channels, output_file):
    """Zips up X, y, and cellTypes.json into a DCL zip file.

    Args:
        X (np.array): raw image
        y (np.array): segmentation mask
        cell_types (list): cell types json
        channels (list): list of channels
    """

    print('Zipping everything up...\n')
    mf = io.BytesIO()
    with zipfile.ZipFile(mf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Write raw image X to X.ome.tiff
        image = io.BytesIO()
        with TiffWriter(image, ome=True) as tif:
            metadata = {
                'axes': 'CZYX',
                'Channel': {'Name': channels}
            }
            tif.write(X, metadata=metadata)
        image.seek(0)
        zf.writestr('X.ome.tiff', image.read())

        # Write segmentation mask y to y.ome.tiff
        segmentation = io.BytesIO()
        with TiffWriter(segmentation, ome=True) as tif:
            tif.write(y, metadata={'axes': 'CZYX'})
        segmentation.seek(0)
        zf.writestr('y.ome.tiff', segmentation.read())

        # Write cellTypes json cell_types_json to cellTypes.json
        cell_types_data = json.dumps(cell_types, indent=2)
        zf.writestr('cellTypes.json', cell_types_data)

        # TODO: Create project through DCL API
        # mf.seek(0)
        # x = requests.post('http://127.0.0.1:5000/api/project',
        #                   data={'images': X, 'labels': y, 'axes': 'CZYX'},
        #                   #   headers={'Content-Type': 'multipart/form-data'},
        #                   )
        # print(x.text)

    with open(output_file, 'wb') as f:
        f.write(mf.getvalue())
        print('Done!')


def main(args):
    if args.recursive:
        for root, dirs, files in os.walk(args.raw_file_path):
            for file in files:
                if file.endswith('.npz'):
                    X, y, cell_types, kept_channels = raw_to_dcl(
                        args.tile_x, args.tile_y, args.tile_center, args.ground_truth, args.marker_positivity, args.clahe, os.path.join(root, file), args.metadata, args.config)
                    dcl_zip(X, y, cell_types, kept_channels, os.path.join(
                        args.output_file, file.replace('.npz', '.zip'))
                    )
    else:
        X, y, cell_types, kept_channels = raw_to_dcl(
            args.tile_x, args.tile_y, args.tile_center, args.ground_truth, args.marker_positivity, args.clahe, args.raw_file_path, args.metadata, args.config)
        dcl_zip(X, y, cell_types, kept_channels, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert input npz into DCL zip')
    parser.add_argument('--tile_x', '-tx')
    parser.add_argument('--tile_y', '-ty')
    parser.add_argument('--tile_center', '-tc')
    parser.add_argument('--ground_truth', '-g', action='store_true')
    parser.add_argument('--marker_positivity', '-mp', action='store_true')
    parser.add_argument('--clahe', '-c', action='store_true')
    parser.add_argument('--recursive', '-r', action='store_true')
    parser.add_argument('raw_file_path', metavar='./raw_path',
                        type=str, help='File path of the raw npz file.')
    parser.add_argument('metadata', metavar='./meta_path',
                        type=str, help='File path of the metadata')
    parser.add_argument('config', metavar='./config_path',
                        type=str, help='File path of the config file')
    parser.add_argument(
        'output_file', metavar='name.zip', type=str, help='Name of the output zip file')
    args = parser.parse_args()
    main(args)
