import os
import sys


def main():
    mode, config_file = None, None

    try:
        mode = sys.argv[1]
        config_file = sys.argv[2]
    except IndexError:
        print('please provide mode and config file')

    try:
        assert mode in ['train']
    except AssertionError:
        print('mode {} not available'.format(mode))

    try:
        assert os.path.isfile('{}'.format(config_file))
    except AssertionError:
        print('config file {} does not exist'.format(config_file))

    if mode == 'train':
        import train
        train.run(config_file)


if __name__ == '__main__':
    main()
