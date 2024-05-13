import os
import re
from collections import Counter
import shutil


def count_last_numbers_in_filenames(directory):
    # 파일 이름에서 숫자 추출을 위한 정규 표현식
    regex = r"\d+\D*$"

    # 파일 이름의 숫자를 저장할 리스트
    numbers = []

    # 주어진 디렉토리의 모든 파일을 탐색
    for filename in os.listdir(directory):
        # 파일 이름에서 정규 표현식에 맞는 부분을 찾기
        match = re.search(regex, filename)
        if match:
            # 숫자 부분만 추출 (마지막 비숫자 문자 제거)
            last_number = re.sub(r'\D', '', match.group())
            if last_number:
                numbers.append(last_number)

    # 숫자들의 빈도 계산
    counter = Counter(numbers)

    return counter


def organize_files_by_last_number(source_directory):
    # 폴더 생성 및 파일 이동을 위한 정규 표현식
    regex = r"\d+\D*$"

    # 1부터 9까지의 폴더 생성
    for i in range(1, 10):
        os.makedirs(os.path.join('/media/hskim/data/Mill_DS/data/class_distribution/', str(i)), exist_ok=True)

    # 파일을 읽고 적절한 폴더로 이동
    for filename in os.listdir(source_directory):
        # 파일 이름에서 마지막 숫자 부분 추출
        match = re.search(regex, filename)
        if match:
            last_number = re.sub(r'\D', '', match.group())
            if last_number and last_number.isdigit():
                last_digit = int(last_number[-1])
                if 1 <= last_digit <= 9:
                    # 원본 파일 경로
                    original_path = os.path.join(source_directory, filename)
                    # 목적지 파일 경로
                    destination_path = os.path.join('/media/hskim/data/Mill_DS/data/class_distribution/', str(last_digit), filename)
                    # 파일 이동
                    shutil.move(original_path, destination_path)
                    print(f"Moved {filename} to folder {last_digit}")

# 사용 예시: 'path/to/your/directory'에 자신의 파일 경로를 입력
directory_path = '/media/hskim/data/Mill_DS/data/Camera_6'
organize_files_by_last_number(directory_path)