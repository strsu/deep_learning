# Document Contrast Enhancement Model

### Description
이 모델은 문서 이미지 내 텍스트 영역은 보존하면서 배경 부분의 Contrast를 균일하게 해주는 딥러닝 모델입니다.
![res4](https://user-images.githubusercontent.com/25381921/148644021-d5bd7b80-faca-45d4-8343-ae26a797b725.png)

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
 
 ![trainSet](https://user-images.githubusercontent.com/25381921/148642778-734947dc-38fc-448b-afe4-6bb637b03d9f.png)

 아래 두 이미지는 contrast를 조정할 영역에 대한 label이다.
 
 ![part](https://user-images.githubusercontent.com/25381921/148642780-8226fcfd-643a-4888-815e-c937cae9bb29.png)
 ![hole](https://user-images.githubusercontent.com/25381921/148642783-8045f6ab-d6eb-4c5e-beb1-2fe5a8170892.png)

### Result
 명암대비 강화를 적용한 이미지의 성능을 평가하기 위해 명암대비가 뚜렷하게 있는 Dataset을 가지고 OCR 인식률을 비교해보았습니다.
 [WEZUT_OCR_Dataset](http://okarma.zut.edu.pl/index.php?id=dataset)

 * 모델 적용 전 후

  #WEZUT_OCR_Dataset
   ![res3](https://user-images.githubusercontent.com/25381921/148643895-cdb2239d-868a-45ce-a32e-65961ace6c53.png)
   ![res4](https://user-images.githubusercontent.com/25381921/148644021-d5bd7b80-faca-45d4-8343-ae26a797b725.png)
  
  #Image From Internet
   ![res1](https://user-images.githubusercontent.com/25381921/148643388-2675c98d-c384-4d93-b2fe-855bbc70b1b9.png)
  
  #Our vs CLAHE
   ![res2](https://user-images.githubusercontent.com/25381921/148643608-ad4629af-6881-4858-9579-34aea1fd2fc0.png)
 
 * 평균 OCR 인식률
 
  WEZUT_OCR_Dataset은 총 176장의 이미지입니다.
   
  | Original | Our | CLAHE |
  | - | - | - |
  | 50% | 90% | 67% |
