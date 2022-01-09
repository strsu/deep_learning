# Defect Detection And Elimination Model

### Description
이 모델은 문서 이미지 내 존재하는 결점(블러, 노이즈, 명암대비)을 탐지하고, 탐지된 결점을 제거해주는 모델입니다.   

![image](https://user-images.githubusercontent.com/25381921/148665379-8410d2b1-ca00-4336-ba04-75aa5870b7fd.png)
<img width="60%" src="https://user-images.githubusercontent.com/25381921/148665379-8410d2b1-ca00-4336-ba04-75aa5870b7fd.png"></img>

selectModel을 통해 이미지 내 결점을 탐지를 합니다. 탐지된 결점이 없다면 곧바로 OCR을 수행합니다.   
탐지된 결점이 1개라면 1개의 결점에 대해 전처리 작업을 수행 후 OCR을 수행합니다.   
탐지된 결점이 2개 이상이라면 모든 경우의 수에 대해 전처리 작업을 수행 후 OCR을 수행하고 나온 예측 정확도 중 가장 높은 예측 정확도의 Text를 선택합니다.

#### ver2
논문 내용과 달리 훈련 데이터는 혼합 결점 이미지(블러+노이즈 등)없이 단일 결점 이미지로 이루어진 데이터셋만 이용해서 훈련을 합니다.

### Prerequisites
 * Python 3.6.12
 * Pytorch 1.7.1
 * CUDA 10.2
 * Anaconda 3
 * OpenCV 3.4.16
 * Tesseract 4, config='--psm 6 --oem 1'

 * [pretrained model](https://drive.google.com/file/d/1R9T5n0tQ90sb8TfmkTVE0s55iXtP1ohZ/view?usp=sharing)

### Used Preprocessing Model
[SRN-Deblur](https://github.com/jiangsutx/SRN-Deblur)

[DnCNN](https://github.com/SaoYan/DnCNN-PyTorch)

[Contrast Enhancement](https://github.com/strsu/sku_deep_learning/tree/main/Contras_Enhancement)

SRN-Deblur, DnCNN은 기존의 논문과 코드를 가져와 사용하였습니다.   
Contrast Enhancement는 직접만든 명암대비 강화모델을 사용하였습니다.

-> 3개의 모델 모두 docVOC Dataset을 기반으로 훈련시켰습니다. 블러와 노이즈는 [imgaug](https://imgaug.readthedocs.io/en/latest/)라이브러리를 이용해 랜덤 수치로 다양한 블러기법, 노이즈기법을 적용한 데이터셋입니다.

### Training Dataset
 wiki에서 가져온 단어를 기반으로 Opencv의 Puttext를 이용해 직접 학습 데이터를 생성하였습니다.
 
 * 훈련 데이터셋 : 7200장
 
 ![image](https://user-images.githubusercontent.com/25381921/148665083-dd522f7c-6530-4a64-ac4d-7a830dbf27bf.png)


### Result
 평가 이미지에 자연스러운 명암대비를 추가하기 어려워 자연스러운 명암대비가 있는 Dataset을 가지고 OCR 인식률을 비교해보았습니다.   
 [WEZUT_OCR_Dataset](http://okarma.zut.edu.pl/index.php?id=dataset)   
 총 176장의 명암대비기반 이미지에 랜덤 블러, 노이즈를 추가해 총 1,235장의 평가 셋을 준비하였습니다.

 * 평가 셋 예시

![image](https://user-images.githubusercontent.com/25381921/148665154-e086d1e2-aeba-4f87-8cf1-8861c57ebb3a.png)

 
 * 평균 OCR 인식률
  
  |   | 평균 OCR 인식률 |
  | :---: | :---: |
  | 전처리 미적용 | 38% |
  | 제안 모델 적용 | 63% |
