# Defect Detection And Elimination Model

### Description
이 모델은 문서 이미지 내 존재하는 결점(블러, 노이즈, 명암대비)을 탐지하고, 탐지된 결점을 제거해주는 모델입니다.

### Prerequisites
 * Python 3.6.12
 * Pytorch 1.7.1
 * CUDA 10.2
 * Anaconda 3
 * OpenCV 3.4.16
 * Tesseract 4, config='--psm 6 --oem 1'

 * [pretrained model](https://drive.google.com/file/d/1R9T5n0tQ90sb8TfmkTVE0s55iXtP1ohZ/view?usp=sharing)

### Training Dataset
 docVOC dataset에 임의로 만든 영역에 따라 문서의 영역을 랜덤한 수치로 contrast를 조절하였습니다.
 

 아래 두 이미지는 contrast를 조정할 영역에 대한 label이다.
 

### Result
 명암대비 강화를 적용한 이미지의 성능을 평가하기 위해 명암대비가 뚜렷하게 있는 Dataset을 가지고 OCR 인식률을 비교해보았습니다.
 [WEZUT_OCR_Dataset](http://okarma.zut.edu.pl/index.php?id=dataset)

 * 모델 적용 전 후
 
 * 평균 OCR 인식률
 
  WEZUT_OCR_Dataset은 총 176장의 이미지입니다.
   
  | Original | Our | CLAHE |
  | - | - | - |
  | 50% | 90% | 67% |