## [sku]deep_learning


> ### Contrast Enhancement

<img width="80%" src="https://user-images.githubusercontent.com/25381921/148673362-5a2f96ba-fe9f-4096-9df2-be7d86c0f0b1.png"></img>

명암대비가 존재하는 이미지에 대해 명암대비 강화를 해주는 모델입니다. 이를 통해 텍스트영역은 손실을 최소화 하면서 배경영역의 Contrast를 균일하게 해줍니다.

***

> ### Preprocessing_SelectModel

이미지에서 블러, 노이즈, 명암대비의 존재여부를 파단해주는 딥러닝 모델입니다.

<img width="80%" src="https://user-images.githubusercontent.com/25381921/148673395-4b4a6cd5-2304-41e1-a3a0-aafc8d72a6ff.png"></img>

다음과 같이 입력 이미지에 대해 각각의 결점에 대한 존재여부를 확률값으로 나타내줍니다. 이때 확률값이 50%를 넘는다면 해당 결점이 존재한다고 판단하게됩니다.

