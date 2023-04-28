# deepcell-label-processing

Scripts for processing and converting raw data to label format and creating DCL projects

## Usage

Currently, the script takes 3 inputs:

- .npz file that contains arrays for X (raw image) and y (segmentation mask) in the dimension order (TYXC)
- Data-registry metadata file
- Dataset configuration file

And outputs:

- .zip file in DeepCell Label format which has:
  - cellTypes.json, which has each type in the "Cell Type Master List" but is otherwise empty
  - X.ome.tiff, the raw data with the names of channels constituting intersection of the "Channel Master List" and the channels specified in the metadata.
  - y.ome.tiff, the segmentation mask

To use the script, run:

```bash
python raw_to_dcl.py [/path/to/raw.npz] [/path/to/metadata.yaml.dvc] [/path/to/config.yaml] [output_name.zip]
```

## TO-DO

- Use dimension order in the metadata file to determine how to reorder X.ome.tiff
- Point to a dvc file instead of an npz file, pull that npz, and then create the project
- Allow user to programatically create a DCL project (ie. with a POST request)
  - This may require changing how DCL creates projects since I think the POST request does not take zip files
- Integrate deepcell-types model to generate embeddings
- More command-line arguments and flexibility, eg. for including ground-truth in cellTypes.json
