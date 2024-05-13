import multiprocessing
import os
import json
import cv2
import numpy as np
from math import prod
import random
import shutil
import time

def save_img(child_pipe):
    while True:
        time.sleep(0.01)
        file_name, img = child_pipe.recv()
        # print(file_name, img.shape)
        if file_name == 'stop':
            child_pipe.close()
            return
        else:
            cv2.imwrite(file_name, img)

def copy_img(child_pipe):
    while True:
        time.sleep(0.01)
        src_file, dst_file = child_pipe.recv()
        if src_file == 'stop':
            child_pipe.close()
            return
        else:
            shutil.copy(src_file, dst_file)


def extract_data():
    data_config = '/home/hskim/Desktop/generative_ai/Gen_AI/config/data_config_16oz_33_00_1.json'
    pre_define = "./predefined_classes.txt"
    camera_config = "/home/hskim/Desktop/generative_ai/Gen_AI/config/camera_config_33_00_00_train.json"

    with open(pre_define, encoding='UTF-8') as f:
        cls = f.readlines()
    cls = [c.replace('\n', '') for c in cls]

    with open(data_config, "r", encoding='UTF-8') as st_json:
        data_json = json.load(st_json)

    original_data_path = data_json['original_data_path']
    data_path = data_json['data_path']
    grid_size = data_json['grid_size']
    grid_count = data_json['grid_count']
    grid_out_count = data_json['grid_out_count']
    grid_padding = data_json['grid_padding']
    grid_stride = data_json['grid_stride']
    multiprocess_core = data_json['multiprocess_core']



    with open(camera_config, "r", encoding='UTF-8') as st_json:
        camera_json = json.load(st_json)

    window_y = grid_count[0] + grid_out_count[0] * 2
    window_x = grid_count[1] + grid_out_count[1] * 2
    stride_y = grid_stride[0]
    stride_x = grid_stride[1]

    os.makedirs(data_path + '/data/Camera_1', exist_ok=True)
    os.makedirs(data_path + '/data/Camera_2', exist_ok=True)
    os.makedirs(data_path + '/data/Camera_3', exist_ok=True)
    os.makedirs(data_path + '/data/Camera_4', exist_ok=True)
    os.makedirs(data_path + '/data/Camera_5', exist_ok=True)
    os.makedirs(data_path + '/data/Camera_6', exist_ok=True)

    ps = []
    for _ in range(multiprocess_core):
        parent_pipe, child_pipe = multiprocessing.Pipe()
        p = multiprocessing.Process(name="child", target=save_img, args=(child_pipe,))
        p.start()
        ps.append([parent_pipe, child_pipe, p])

    ps_index = 0
    for dir_path, parse_type in original_data_path:
        if parse_type == 'true_false':
            dir_path += '/labels'
        elif parse_type == 'only_true':
            print(dir_path)
            dir_path += '/crops'
        elif parse_type == 'only_false':
            dir_path += '/labels'

        for f in os.listdir(dir_path):
            if 'classes.txt' in f:
                continue
            if 'false' in parse_type:
                label_file = dir_path + '/' + f
                try:
                    with open(label_file, 'r', encoding='UTF-8') as f:
                        labels = f.readlines()
                except:
                    print(label_file)
                    continue
                if len(labels) == 0:
                    continue
                labels = [lb.replace('\n', '').split(' ') for lb in labels]


                img_file = label_file.replace('.txt', '.png').replace('/labels/', '/crops/')
            else:
                img_file = dir_path + '/' + f
            img_name = img_file.replace('.png', '').split('/')[-1]
            img = cv2.imread(img_file)
            if img is None:
                print('data_preprocess.py :: not exist img file', img_file)
                continue
            img_h, img_w, img_c = img.shape

            new_img_h = img_h + grid_padding[0] * grid_size * 2
            new_img_w = img_w + grid_padding[1] * grid_size * 2

            new_img = np.zeros((new_img_h, new_img_w, img_c), dtype=img.dtype)
            new_img[grid_padding[0] * grid_size : grid_padding[0] * grid_size + img_h, grid_padding[1] * grid_size : grid_padding[1] * grid_size + img_w] = img

            # grid line
            # for _x in range(1, new_img_w // grid_size):
            #     cv2.line(new_img, (grid_size * _x, 0), (grid_size * _x, new_img_h), (0, 0, 255), 1)
            #
            # for _y in range(1, new_img_h // grid_size):
            #     cv2.line(new_img, (0, grid_size * _y), (new_img_w, grid_size * _y), (0, 0, 255), 1)

            if 'only_true' in parse_type:
                for _x in range(0, (new_img_w // grid_size - window_x // 2) // stride_x):
                    for _y in range(0, (new_img_h // grid_size - window_y // 2) // stride_y):
                        crop_img = new_img[(_y * stride_y) * grid_size: (_y * stride_y + window_y) * grid_size, (_x * stride_x) * grid_size: (_x * stride_x + window_x) * grid_size]
                        f_name = data_path + '/data/' + '_'.join(img_name.split('_')[1:3]) + '/' + dir_path.split('/')[-2] + '_' + img_name + '_' + str(_x) + '_' + str(_y) + '(' + str(cls.index('정상')) + ')' + str(prod(grid_count)) + '.png'
                        ps[ps_index][0].send([f_name, crop_img])
                        ps_index += 1
                        if ps_index >= multiprocess_core:
                            ps_index = 0
            else:
                grid_cls = np.zeros((new_img_h // grid_size, new_img_w // grid_size, len(cls)))

                for lb in labels:
                    c, cx, cy, w, h = lb
                    c = int(c)
                    cx = float(cx) * img_w
                    cy = float(cy) * img_h
                    gx = int(cx // grid_size)
                    gy = int(cy // grid_size)
                    grid_cls[gy + grid_padding[0], gx + grid_padding[1], c] = 1



                '''
                _x, _y는 이미지0,0기준부터 시작임(grid_out_count제외)
                '''
                for _x in range(0, grid_cls.shape[1] - window_x + 1):
                    for _y in range(0, grid_cls.shape[0] - window_y + 1):

                        if [_x, _y] not in camera_json['_'.join(img_name.split('_')[1:3])]:
                            continue


                        crop_img = new_img[(_y - grid_out_count[0]) * grid_size : (_y - grid_out_count[0] + window_y) * grid_size, (_x - grid_out_count[1]) * grid_size : (_x - grid_out_count[1] + window_x) * grid_size]
                        merge_cls = np.sum(np.sum(grid_cls[_y:_y + window_y, _x:_x + window_x], axis=0), axis=0).astype(np.int64)

                        merge_sum = np.sum(merge_cls)
                        #애매함 하나라도있으면 pass임
                        if merge_cls[cls.index('애매함')] != 0:
                            continue

                        if merge_cls[cls.index('기타')] != 0:
                            continue

                        lbs = ''
                        for _c in range(len(cls)):
                            if merge_cls[_c] != 0:
                                lbs += '(' + str(_c) + ')' + str(merge_cls[_c])

                        chk = True
                        if merge_sum - merge_cls[cls.index('QR코드')] > 0:
                            chk = False
                        if chk and lbs == '':
                            lbs += '(' + str(cls.index('정상')) + ')' + str(prod(grid_count))

                        if 'true' in parse_type:
                            if chk:
                                f_name = data_path + '/data/' + '_'.join(img_name.split('_')[1:3]) + '/' + dir_path.split('/')[-2] + '_' + img_name + '_' + str(_x) + '_' + str(_y) + lbs + '.png'
                                ps[ps_index][0].send([f_name, crop_img])
                                ps_index += 1
                                if ps_index >= multiprocess_core:
                                    ps_index = 0
                        if 'false' in parse_type:
                            if not chk:
                                f_name = data_path + '/data/' + '_'.join(img_name.split('_')[1:3]) + '/' + dir_path.split('/')[-2] + '_' + img_name + '_' + str(_x) + '_' + str(_y) + lbs + '.png'
                                ps[ps_index][0].send([f_name, crop_img])
                                ps_index += 1
                                if ps_index >= multiprocess_core:
                                    ps_index = 0

    for pp, pc, p in ps:
        pp.send(['stop', None])
        p.join()
    time.sleep(10)
    for pp, pc, p in ps:
        try:
            pp.close()
        except:
            pass
        try:
            pc.close()
        except:
            pass

def split_train_valid():
    data_config = "/home/hskim/Desktop/generative_ai/Gen_AI/config/data_config_16oz_33_00_1.json"
    split_config = "/home/hskim/Desktop/generative_ai/Gen_AI/config/slplit_config_1.json"

    with open(data_config, "r", encoding='UTF-8') as st_json:
        data_json = json.load(st_json)

    data_path = data_json['data_path']

    with open(split_config, "r", encoding='UTF-8') as st_json:
        split_json = json.load(st_json)

    data_split = split_json['data_split']
    multiprocess_core = split_json['multiprocess_core']

    os.makedirs(data_path + '/train/Camera_1', exist_ok=True)
    os.makedirs(data_path + '/train/Camera_2', exist_ok=True)
    os.makedirs(data_path + '/train/Camera_3', exist_ok=True)
    os.makedirs(data_path + '/train/Camera_4', exist_ok=True)
    os.makedirs(data_path + '/train/Camera_5', exist_ok=True)
    os.makedirs(data_path + '/train/Camera_6', exist_ok=True)
    os.makedirs(data_path + '/valid/Camera_1', exist_ok=True)
    os.makedirs(data_path + '/valid/Camera_2', exist_ok=True)
    os.makedirs(data_path + '/valid/Camera_3', exist_ok=True)
    os.makedirs(data_path + '/valid/Camera_4', exist_ok=True)
    os.makedirs(data_path + '/valid/Camera_5', exist_ok=True)
    os.makedirs(data_path + '/valid/Camera_6', exist_ok=True)

    dic_path = {}
    for cam_num in range(1, 7):
        if 'Camera_' + str(cam_num) not in dic_path:
            dic_path['Camera_' + str(cam_num)] = {}
        for f in os.listdir(data_path + '/data/Camera_' + str(cam_num)):
            f_cls = '_'.join(sorted([_c.split(')')[0] for _c in f.replace('.png', '').split('(')[1:]]))
            if f_cls not in dic_path['Camera_' + str(cam_num)]:
                dic_path['Camera_' + str(cam_num)][f_cls] = []
            dic_path['Camera_' + str(cam_num)][f_cls].append(data_path + '/data/Camera_' + str(cam_num) + '/' + f)

    ps = []
    ps_index = 0
    for _ in range(multiprocess_core):
        parent_pipe, child_pipe = multiprocessing.Pipe()
        p = multiprocessing.Process(name="child", target=copy_img, args=(child_pipe,))
        p.start()
        ps.append([parent_pipe, child_pipe, p])

    for dic_key in dic_path.keys():
        for key in dic_path[dic_key].keys():
            p = dic_path[dic_key][key]
            random.shuffle(p)
            split_pos = int(len(p) * data_split)
            for f in p[:split_pos]:
                ps[ps_index][0].send([f, f.replace('/data/', '/train/')])
                ps_index += 1
                if ps_index >= multiprocess_core:
                    ps_index = 0
            for f in p[split_pos:]:
                ps[ps_index][0].send([f, f.replace('/data/', '/valid/')])
                ps_index += 1
                if ps_index >= multiprocess_core:
                    ps_index = 0
    for pp, pc, p in ps:
        pp.send(['stop', None])
        p.join()
    time.sleep(10)
    for pp, pc, p in ps:
        try:
            pp.close()
        except:
            pass
        try:
            pc.close()
        except:
            pass

def augment_option(model_config, img):
    flip = model_config['flip']
    rot = model_config['rot']
    if np.random.random() > 0.5:
        cv2.flip(img, flip)#좌우반전 0이면 상하반전임

    if np.random.random() > 0.5:
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 이미지의 중심을 중심으로 이미지를 45도 회전합니다.
        M = cv2.getRotationMatrix2D((cX, cY), np.random.choice([-1, 1]) * np.random.randint(1, rot + 1), 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    return img

def gen_data(train=False):
    data_config = "/home/hskim/Desktop/generative_ai/Gen_AI/config/data_config_16oz_33_00_00_1.json"
    model_config = "/home/hskim/Desktop/generative_ai/Gen_AI/config/model_config_side.json"
    camera_config ="/home/hskim/Desktop/generative_ai/Gen_AI/config/camera_config_33_00_00_train.json"
    pre_define = "./predefined_classes.txt"

    with open(pre_define, encoding='UTF-8') as f:
        cls = f.readlines()
    cls = [c.replace('\n', '') for c in cls]


    with open(data_config, "r", encoding='UTF-8') as st_json:
        data_json = json.load(st_json)

    data_path = data_json['data_path']



    with open(model_config, "r", encoding='UTF-8') as st_json:
        model_json = json.load(st_json)

    augment = model_json['augment']
    train_cls = model_json['train_cls']
    cam = model_json['cam']
    train_len = model_json['train_len']
    train_set = model_json['train_set']
    batch_size = model_json['batch_size']


    with open(camera_config, "r", encoding='UTF-8') as st_json:
        camera_json = json.load(st_json)


    target_path = 'train'
    if not train:
        target_path = 'valid'

    # 정상/세척/스크래치/파손/보풀
    # cls / [pos, value]
    if train_cls[0] == -1:
        dic_path = {'true':[], 'false':[]}
        for cam_num in cam:
            for f in os.listdir(data_path + '/' + target_path + '/Camera_' + str(cam_num)):
                if int(f.split('Camera_')[1][0]) in cam:
                    ls_pos = list(map(int, f.split('Camera_' + str(cam_num) + '_16_16_')[1].split('(')[0].split('_')))
                    if ls_pos in camera_json['Camera_' + str(cam_num)]:
                        chk = 0
                        for key in f.replace('.png', '').split('(')[1:]:
                            if str(cls.index('정상')) + ')' in key:
                                chk = 1
                            elif str(cls.index('QR코드')) + ')' in key:
                                chk = 1
                            elif str(cls.index('기타')) + ')' in key:
                                chk = 0
                                break
                            else:
                                chk = 2
                        if chk == 1:#정상
                            dic_path['true'].append(data_path + '/' + target_path + '/Camera_' + str(cam_num) + '/' + f)
                        elif chk == 2:#불량클래스
                            dic_path['false'].append(data_path + '/' + target_path + '/Camera_' + str(cam_num) + '/' + f)

        random.shuffle(dic_path['true'])
        random.shuffle(dic_path['false'])

        for i in range(len(dic_path['false'])):
            img = cv2.imread(dic_path['true'][i])
            if img is None:
                print(dic_path['true'][i])
            if augment:
                if train:
                    img = augment_option(model_json, img)

            # yield img / 255., [0.]

            img = cv2.imread(dic_path['false'][i])
            if img is None:
                print(dic_path['false'][i])
            if augment:
                if train:
                    img = augment_option(model_json, img)

            # yield img / 255., [1.]

    else:
        dic_path = {'true':[], 'false':[]}

        true_cls = []
        for key in train_set.keys():
            if train_set[key][0] == 0:
                true_cls.append(key)

        for cam_num in cam:
            for f in os.listdir(data_path + '/' + target_path + '/Camera_' + str(cam_num)):
                if int(f.split('Camera_')[1][0]) in cam:
                    if '(' not in f:
                        continue
                    try:
                        ls_pos = list(map(int, f.split('(')[0].split('_')[-2:]))
                    except:
                        print(f)
                    if ls_pos in camera_json['Camera_' + str(cam_num)]:
                        chk = 0
                        for key in f.replace('.png', '').split('(')[1:]:
                            if int(key.split(')')[0]) not in train_cls:
                                chk = 3
                                break
                            if key.split(')')[0] in true_cls:
                                if chk == 0:
                                    chk = 1
                            else:
                                chk = 2

                        if chk == 1:#정상
                            dic_path['true'].append(data_path + '/' + target_path + '/Camera_' + str(cam_num) + '/' + f)
                        elif chk == 2:#불량클래스
                            dic_path['false'].append(data_path + '/' + target_path + '/Camera_' + str(cam_num) + '/' + f)


        random.shuffle(dic_path['true'])
        random.shuffle(dic_path['false'])
        dic_false = {}
        for f in dic_path['false']:
            for false_c in f.replace('.png', '').split('(')[1:]:
                false_c = false_c.split(')')[0]
                if str(false_c) in true_cls:
                    continue
                if false_c not in dic_false:
                    dic_false[false_c] = []
                dic_false[false_c].append(f)
        print()
        for key in dic_false.keys():
            random.shuffle(dic_false[key])
            print('key : ', key, len(dic_false[key]))
        print()
        i = 0
        if train:
            data_num = 16
        else:
            data_num = 1
        while i < batch_size*data_num:
            img = cv2.imread(np.random.choice([f for f in dic_path['true'] if '16_16_' + '_'.join(list(map(str, camera_json['Camera_' + str(cam[0])][i % len(camera_json['Camera_' + str(cam[0])])]))) in f]))
            if img is None:
                print(dic_path['true'][i])
                continue
            if augment:
                if train:
                    img = augment_option(model_json, img)

            lb = np.zeros(train_len, dtype=np.float32)
            yield img / 255., lb

            f = np.random.choice(dic_false[list(dic_false.keys())[i % len(dic_false.keys())]])
            img = cv2.imread(f)
            if img is None:
                print(dic_path['false'][i])
                continue
            if augment:
                if train:
                    img = augment_option(model_json, img)

            lb = np.zeros(train_len, dtype=np.float32)
            try:
                for l in f.replace('.png', '').split('(')[1:]:
                    c, v = list(map(int, l.split(')')))
                    if str(c) in true_cls:
                        continue
                    lb[train_set[str(c)][0]] += train_set[str(c)][1] * v
            except:
                k = 10
            lb = np.where(lb > 4, 4, lb)
            if np.sum(lb[1:]) > 0:
                lb[0] = 1

            lb = np.where(lb > 0, lb/4, 0)

            yield img / 255., lb


            i += 1

extract_data()
split_train_valid()
gen_data(train=True)
k=10