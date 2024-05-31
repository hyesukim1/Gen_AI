import json

# 안써도 될 것 같음
class ConfigManager():
    """ 설정 관리 클래스 """

    def __init__(self, main_conf_path):
        """
         ConfigManager 클래스의 생성자입니다.

        :param main_conf_path: 주 설정 파일 경로
        """
        self.main_conf = main_conf_path

    def read_config(self, config_file):
        """
        JSON 형식의 설정 파일을 읽어서 파이썬 객체로 반환합니다.

        :param config_file: 읽을 설정 파일 경로
        :return: 설정 파일에 대응하는 파이썬 객체
        """
        with open(config_file) as f:
            return json.load(f)

    def check_list_of_config(self):
        """
        주 설정 파일에서 데이터 설정과 모델 설정을 확인하고 반환합니다.

        :return: 데이터 설정과 모델 설정을 담은 튜플
        """
        main_conf = self.read_config(self.main_conf)
        data_conf = self.read_config(main_conf['data_config'])
        model_conf = self.read_config(main_conf['model_config'])
        return data_conf, model_conf