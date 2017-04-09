import argparse
import numpy as np
import cv2
import keras.backend as K
from vin import vin_model, get_layer_output
from utils import process_map_data

def get_action(a):
    if a == 0: return -1, -1
    if a == 1: return  0, -1
    if a == 2: return  1, -1
    if a == 3: return -1,  0
    if a == 4: return  1,  0
    if a == 5: return -1,  1
    if a == 6: return  0,  1
    if a == 7: return  1,  1
    return None

def find_goal(m):
    return np.argwhere(m.max() == m)[0][::-1]

def predict(im, pos, model, k):
    im_ary = np.array([im]).transpose((0, 2, 3, 1)) if K.image_dim_ordering() == 'tf' else np.array([im])
    res = model.predict([im_ary,
                         np.array([pos])])

    action = np.argmax(res)
    reward = get_layer_output(model, 'reward', im_ary)
    value = get_layer_output(model, 'value{}'.format(k), im_ary)
    reward = np.reshape(reward, im.shape[1:])
    value = np.reshape(value, im.shape[1:])

    return action, reward, value

def main():
    parser = argparse.ArgumentParser(description='VIN')
    parser.add_argument('--data', '-d', type=str, default='./data/map_data.pkl',
                        help='Path to map data generated with script_make_data.py')
    parser.add_argument('--model', '-m', type=str, default='vin_model_weights.h5',
                        help='Model from given file')
    args = parser.parse_args()

    k = 20
    train, test = process_map_data(args.data)
    model = vin_model(l_s=test[0].shape[2], k=k)
    model.load_weights(args.model)

    for d in zip(*test):
        im = d[0]
        pos = d[1]
        action, reward, value = predict(im, pos, model, k)

        path = [tuple(pos)]
        for _ in range(30):
            if im[1][pos[1], pos[0]] == 1:
                break
            action, _, _ = predict(im, pos, model, k)
            dx, dy = get_action(action)
            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy
            path.append(tuple(pos))

        test_img = cv2.cvtColor(im[0], cv2.COLOR_GRAY2BGR)
        goal = find_goal(im[1])

        for s in path:
            cv2.rectangle(test_img, (s[0], s[1]), (s[0], s[1]), (1, 0, 0), -1)
        cv2.rectangle(test_img, (path[0][0], path[0][1]), (path[0][0], path[0][1]), (0, 1, 1), -1)
        cv2.rectangle(test_img, (goal[0], goal[1]), (goal[0], goal[1]), (0, 0, 1), -1)
        cv2.imshow("image", cv2.resize(255 - test_img * 255, (300, 300), interpolation=cv2.INTER_NEAREST))
        cv2.imshow("reward", cv2.resize(reward, (300, 300), interpolation=cv2.INTER_NEAREST))
        cv2.imshow("value", cv2.resize(value / 80, (300, 300), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
