import joblib
import numpy as np

def process_map_data(path):
    data = joblib.load(path)

    im_data = data['im']
    value_data = data['value']
    state_data = data['state']
    label_data = np.array([np.eye(1, 8, l)[0] for l in data['label']])

    num = im_data.shape[0]
    num_train = num - num / 5

    im_train = np.concatenate((np.expand_dims(im_data[:num_train], 1),
                               np.expand_dims(value_data[:num_train], 1)),axis=1).astype(dtype=np.float32)
    state_train = state_data[:num_train]
    label_train = label_data[:num_train]

    im_test = np.concatenate((np.expand_dims(im_data[num_train:], 1),
                              np.expand_dims(value_data[num_train:], 1)),axis=1).astype(dtype=np.float32)
    state_test = state_data[num_train:]
    label_test = label_data[num_train:]

    return (im_train, state_train, label_train), (im_test, state_test, label_test)
