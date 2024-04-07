from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import copy
import requests
from io import BytesIO
from log import Logger
import re


hanja_ptrn = re.compile(r'[一-龥]')
kor_ptrn = re.compile(r'[가-힣]')
kor_vowel_consonant_ptrn = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]]')
eng_ptrn = re.compile(r'[a-zA-Z]')


ocr = PaddleOCR(lang="korean")
log = Logger.get_instance()


# def slice_image_vertically(image_path, gap):
      
#     # 이미지 로드
#     image = Image.open(image_path)
#     width = image.size[0]   # 이미지 가로
#     height = image.size[1]  # 이미지 세로
    
#     cuted_pix=[]
#     # 이미지 잘라서 저장
#     for i in range(0, height, gap):
#         box = (0, i, width, min(i + gap, height))   # 맨 끝부분
#         cuted_pix.append(min(i + gap, height))
#         image_gap = image.crop(box)
#         image_gap.save(os.path.join(folder_name, f"{i}.jpg"))
    
#     return height # 높이값 반환


def OCR_image(images_array, gap:int, ocr_engine): 
    """
    images_array: list[nd.array]
    """
    # 텍스트랑 좌표 1:1로 매칭시켜서 넣을 리스트
    bbox_sequence=[]
    for cnt, image_array in enumerate(images_array):
        # ocr 수행
        try:
            # detection(bounding box)만 진행.
            result=ocr_engine.ocr(image_array, det=True, rec=False, cls=False) 
        except ValueError as val_error:
            log.error("[ERROR] value error in lib. Must modify in paddleocr line 681.")
        
        # 잘린 사진에서 글이 없을 수 있음.
        if not result or not result[0]:
            continue

        # 바운딩박스 좌표들을 갭에 맞춰서 리스트에 추가.    
        for line in result[0]:                    
            bbox_sequence.append([[ax[0], ax[1]+gap*cnt] for ax in line])

    return bbox_sequence     #바운딩 박스 좌표랑 문자가 튜플로 묶인 리스트 반환


# 이분탐색
def find_small_largest(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return arr[mid-1] # 일부로 하나 전까지만 자름.
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # while문 끝나고 났을때 right가 더 작은 값이 되어있음
    if(arr[right]<target):
        return arr[right]
    else:
        return arr[right-1]
    


# 이미지 자를 임계값 픽셀 찾기
def char_pix_extract(image_path, ocr_engine):
    """
    image_path: img tag src attribute. e.g link of img.

    ocr_engine: OCREngine instance type
    """

    # slice_image_vertically 가져옴.
    # 이미지 로드
    if 'http' in image_path:
        res = requests.get(image_path)
        image = Image.open(BytesIO(res.content))
    else:
        image = Image.open(image_path)

    
    width = image.size[0]   # 이미지 가로
    height = image.size[1]  # 이미지 세로

    if height < 2000:
        return [0, height]
    
    cropped_images_2000 = []
    cropped_images_1700 = []
    gaps = (1700, 2000) # 최소공배수 34000.
    for gap in gaps:
        # 이미지 잘라서 저장
        if gap == 2000:
            for i in range(0, height, gap):
                box = (0, i, width, min(i + gap, height))   # 맨 끝부분            
                image_gap = np.asarray(image.crop(box), dtype='uint8')
                cropped_images_2000.append(image_gap)
        elif gap == 1700:
            for i in range(0, height, gap):
                box = (0, i, width, min(i + gap, height))   # 맨 끝부분            
                image_gap = np.asarray(image.crop(box), dtype='uint8')
                cropped_images_1700.append(image_gap)

    boundary=[]
    cut_candidate=set()
    for cut_height in gaps: 
        if cut_height == 1700:       
            result = OCR_image(cropped_images_1700, cut_height, ocr_engine)
            cropped_images_1700.clear()
        elif cut_height == 2000:       
            result = OCR_image(cropped_images_2000, cut_height, ocr_engine)
            cropped_images_2000.clear()
        for bounding_box in result:
            top= int(min([bounding_box[i][1] for i in range(4)]))
            bottom=int(max([bounding_box[i][1] for i in range(4)]))
            boundary.append((top,bottom))   # 문자 있는 구간 위아래 기준

            # 위에서 10, 아래에서 10 자르기.
            for i in range(0, 11, 5):
                cut_candidate.add(top-i)
                cut_candidate.add(bottom+i)

        # 문자 구간사이에 들어가는 후보 제거
        to_remove = cut_candidate.copy()
        for candidate in to_remove:
            for pair in boundary:
                if candidate > pair[0] and candidate < pair[1]:
                    cut_candidate.remove(candidate)
                    break

    cut_candidate = sorted(cut_candidate)
        
    result_pix_set = set()
    print("이미지 길이:", height)
    
    cut_line = 2000
    # print("후보:", cut_candidate)
    while cut_line < height:
        new_cut = find_small_largest(cut_candidate, cut_line)
        # 새롭게 찾은게 기준선보다 크거나 같으면, 혹은 이미 있었던 선이면 기준선으로 넣어줘야함.
        if new_cut >= cut_line or new_cut in result_pix_set:
            result_pix_set.add(cut_line)
            cut_line+=2000        
        else:
            # 새롭게 찾은게 기준선보다 작고, 만약 진짜 처음 나온 선이면
            result_pix_set.add(new_cut)
            cut_line = new_cut + 2000
    
    result_pix_set.add(0) # 처음 픽셀 높이 넣어주기.
    result_pix_set.add(height) # 맨 마지막 픽셀 (height) 넣어주기
        
    return sorted(result_pix_set)


# 찾은 임계값 기준으로 이미지 자르기
def last_image_cut(image_path, folder_name, cut_pix_list):
    # 잘린 이미지 저장할 폴더 생성
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # 이미지 로드
    if 'http' in image_path:
        res = requests.get(image_path)
        image = Image.open(BytesIO(res.content))
    else:
        image = Image.open(image_path)

    width = image.size[0]   # 이미지 가로
    
    for i in range(len(cut_pix_list)-1):
        box = (0, cut_pix_list[i], width, cut_pix_list[i+1])
        croped_image = image.crop(box)
        croped_image.save(os.path.join(folder_name, f"{i}.jpg"))


def last_OCR_image(folder_name, cut_pix_list, ocr_engine):
    image_list = os.listdir(folder_name)
    # 이미지 잘린 순서대로 정렬
    image_list = sorted(image_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 텍스트랑 좌표 1:1로 매칭시켜서 넣을 리스트
    ocr_sequence=[]
    for image_index, cut_line in zip(image_list, cut_pix_list[:-1]):
        image_path=os.path.join(folder_name, image_index)
        # ocr 수행
        result=ocr_engine.ocr(image_path, cls=True)

        if not result or not result[0]:
            continue
        
        # 픽셀을 나눠놨기 때문에 result의 바운딩 박스 y좌표에 gap을 더해줘야함
        for line in result[0]:
            # 글자 이상한 것 체크.
            exist_kor_eng = eng_ptrn.search(line[1][0]) and kor_ptrn.search(line[1][0])
            exist_hanja = hanja_ptrn.search(line[1][0])
            exist_kor_vowel_consonant = kor_vowel_consonant_ptrn.search(line[1][0])

            if exist_hanja or exist_kor_vowel_consonant:
                continue
            elif exist_kor_eng:
                if line[1][1] < 0.85:
                    continue                        

            for ax in line[0]:
                ax[1]+=cut_line
        # 바운딩박스 좌표랑 문자를 dict으로 묶어서 리스트에 추가
            ocr_sequence.append({'text':line[1][0], "bbox":line[0]})
        
        # 임시로 만들었던 폴더 제거
    if os.path.exists(folder_name):
        for file_name in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        os.rmdir(folder_name)

    return ocr_sequence     #바운딩 박스 좌표랑 문자가 튜플로 묶인 리스트 반환

