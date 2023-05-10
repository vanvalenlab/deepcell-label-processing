from tifffile import TiffWriter

import argparse
import io
import json
import requests
import zipfile

import utils


def raw_to_dcl(tile_x, tile_y, file_path, metadata, config):
    """  """

    print('Loading raw file...\n')
    X, y = utils.load_raw(file_path)

    print('Parsing config file...\n')
    kept_channels = utils.parse_kept_channels(config)

    print('Parsing metadata file...\n')
    channel_indices, channels = utils.parse_metadata(metadata, kept_channels)

    print('Making cellTypes.json...\n')
    cell_types = utils.make_empty_cell_types()

    print('Making X.ome.tiff...\n')
    X_processed = utils.equalize_adapthist(
        utils.normalize_raw(utils.reshape_X(X, channel_indices)))

    print('Making y.ome.tiff...\n')
    y_processed = utils.reshape_y(y)

    if tile_x and tile_y:
        print("Tiling X and y...")
        X_processed = utils.tile_and_stack_array(
            X_processed, int(tile_x), int(tile_y))
        y_processed = utils.tile_and_stack_array(
            y_processed, int(tile_x), int(tile_y))

    return X_processed, y_processed, cell_types, channels


def dcl_zip(X, y, cell_types, channels):
    """ """

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

    # mf.seek(0)
    # x = requests.post('http://127.0.0.1:5000/api/project',
    #                   data={'images': X, 'labels': y, 'axes': 'CZYX'},
    #                   #   headers={'Content-Type': 'multipart/form-data'},
    #                   )
    # print(x.text)

    with open(args.output_file, 'wb') as f:
        f.write(mf.getvalue())
        print('Done!')


def main(args):
    X, y, cell_types, kept_channels = raw_to_dcl(
        args.tile_x, args.tile_y, args.raw_file_path, args.metadata, args.config)
    dcl_zip(X, y, cell_types, kept_channels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert input npz into DCL zip')
    parser.add_argument('--tile_x', '-tx')
    parser.add_argument('--tile_y', '-ty')
    parser.add_argument('raw_file_path', metavar='./path/to/file',
                        type=str, help='File path of the raw npz file.')
    parser.add_argument('metadata', metavar='./path/to/file',
                        type=str, help='File path of the metadata')
    parser.add_argument('config', metavar='./path/to/file',
                        type=str, help='File path of the config file')
    parser.add_argument(
        'output_file', metavar='name.zip', type=str, help='Name of the output zip file')
    args = parser.parse_args()
    main(args)
