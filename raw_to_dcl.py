from tifffile import TiffWriter
import yaml

import argparse
import io
import json
import subprocess
import zipfile

import utils


def raw_to_dcl(file_path, metadata, config):
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
    X_processed = utils.normalize_raw(utils.reshape_X(X, channel_indices))

    print('Making y.ome.tiff...\n')
    y_processed = utils.reshape_y(y)

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

    with open(args.output_file, 'wb') as f:
        f.write(mf.getvalue())
        print('Done!')


def main(args):

    subprocess.run(
        ['dvc', 'pull', args.metadata],
        capture_output=True,
        encoding='utf-8',
        check=True,
    )
    with open(args.metadata, 'r') as stream:
        metadata = yaml.safe_load(stream)
        file_name = metadata['outs']['path']
    X, y, cell_types, kept_channels = raw_to_dcl(
        file_name, args.metadata, args.config)
    dcl_zip(X, y, cell_types, kept_channels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert input npz into DCL zip')
    parser.add_argument('metadata', metavar='./path/to/file',
                        type=str, help='File path of the dvc metadata')
    parser.add_argument('config', metavar='./path/to/file',
                        type=str, help='File path of the config file')
    parser.add_argument(
        'output_file', metavar='[name.zip]', type=str, help='Name of the output zip file')
    args = parser.parse_args()
    main(args)
