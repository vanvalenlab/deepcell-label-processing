from thefuzz import process
import yaml
import constants

if __name__ == '__main__':
    with open('./configs/meta_hollmann.yaml', 'r') as stream:
        channel_dict = {'channels_to_keep': {}}
        metadata = yaml.safe_load(stream)
        for channel in metadata['meta']['sample']['channels']:
            channel_name = channel['target']
            channel_dict['channels_to_keep'][channel_name] = process.extractOne(
                channel_name, constants.MASTER_CHANNELS)[0]
    with open('./configs/config_hollmann.yaml', 'w') as f:
        yaml.dump(channel_dict, f)
