# Document Contrast Enhancement Model

### Description
이 프로젝트는 문서 이미지 내 존재하는 그림자, 광원 부분을 제거하기 위해 텍스트 영역은 보존하면서 배경 부분을 모두 비슷한 색상을 갖도록 해주는 딥러닝 모델입니다.

### Prerequisites
 * Python 3.6.12
 * Pytorch 1.7.1
 * CUDA 10.2
 * Anaconda 3
 * OpenCV 3.4.16
 * Tesseract 4 - 

### Training Dataset
 docVOC dataset에 임의로 만든 영역에 따라 문서의 영역을 랜덤한 수치로 contrast를 조절하였다.
![trainSet](https://user-images.githubusercontent.com/25381921/148642778-734947dc-38fc-448b-afe4-6bb637b03d9f.png)

 아래 두 이미지는 contrast를 조정할 영역에 대한 label이다.
![part](https://user-images.githubusercontent.com/25381921/148642780-8226fcfd-643a-4888-815e-c937cae9bb29.png)
![hole](https://user-images.githubusercontent.com/25381921/148642783-8045f6ab-d6eb-4c5e-beb1-2fe5a8170892.png)

### Result
 명암대비 강화를 적용한 이미지의 성능을 평가하기 위해 명암대비가 뚜렷하게 있는 Dataset을 가지고 OCR 인식률을 비교해보았다.
 [WEZUT_OCR_Dataset] (http://okarma.zut.edu.pl/index.php?id=dataset)

 * 모델 적용 전 후
   ![res1](https://user-images.githubusercontent.com/25381921/148643388-2675c98d-c384-4d93-b2fe-855bbc70b1b9.png)
 
 * OCR 인식률
  | Original | our | CLAHE |
  | - | - | - |
  | 50% | 90% | 67% |
