import argparse

from keras.optimizers import Adam
import keras.backend as K
from vin import vin_model
from utils import process_map_data

def main():
    parser = argparse.ArgumentParser(description='train vin model')
    parser.add_argument('--data', '-d', type=str, default='./data/map_data.pkl',
                        help='Path to map data generated with script_make_data.py')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    args = parser.parse_args()

    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train, test = process_map_data(args.data)
    model = vin_model(l_s=train[0].shape[2], k=20)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit([train[0].transpose((0, 2, 3, 1)) if K.image_dim_ordering() == 'tf' else train[0],
               train[1]],
              train[2],
              batch_size=args.batchsize,
              nb_epoch=args.epoch)

    with open('vin_model_structure.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('vin_model_weights.h5', overwrite=True)

if __name__ == "__main__":
    main()
