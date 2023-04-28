# deepcell-label-processing

Scripts for processing and converting raw data to label format and creating DCL projects

## Usage

Currently, the script takes 3 inputs:

- .npz file that contains arrays for X (raw image) and y (segmentation mask) in the dimension order (TYXC)
- Data-registry metadata file
- Dataset configuration file

And outputs:

- .zip file in DeepCell Label format which constitutes
  - cellTypes.json, which has each type in the "Cell Type Master List" but is otherwise empty
  - X.ome.tiff, the raw data with the names of channels constituting intersection of the "Channel Master List" and the channels specified in the metadata.
  - y.ome.tiff, the segmentation mask

To use the script, run

```bash
python raw_to_dcl.py [/file/path/to/raw.npz] [/path/to/metadata.yaml.dvc] [/path/to/config.yaml] [output_name.zip]
```
