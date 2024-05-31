import json
import matplotlib.pyplot as plt

def read_config(config_file):
    """
    JSON 형식의 설정 파일을 읽어서 파이썬 객체로 반환합니다.

    :param config_file: 읽을 설정 파일 경로
    :return: 설정 파일에 대응하는 파이썬 객체
    """
    with open(config_file) as f:
        return json.load(f)

def show_images(ori_imgs, model_imgs, data_num=5, image_size=128):
    original_images = ori_imgs
    model_output_images = model_imgs

    f, a = plt.subplots(2, data_num, figsize=(data_num, 2))

    for i in range(data_num):

        a[0][i].imshow(original_images[i], cmap='gray')
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())

        a[1][i].imshow(model_output_images[i], cmap='gray')
        a[1][i].set_xticks(());
        a[1][i].set_yticks(())

    plt.show()





