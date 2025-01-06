# [KCC2024]BiFPN의 적용으로 RetinaFace 모델의 얼굴인식 문제 해결 및 BiRetina 개발  
### 논문 [BiFPN의 적용으로 RetinaFace 모델의 얼굴인식 문제 해결 및 BiRetina 개발.pdf](https://github.com/user-attachments/files/15525930/biRetina_.pdf)




BiFPN의 적용으로 RetinaFace 모델의 얼굴인식 문제 해결 및 BiRetina 개발  

# 요약
본 연구는 고정밀 얼굴 감지를 목표로 하며, 기존 RetinaFace 모델의 성능 한계를 극복하기 위한 새로운 접근법을 제안한다.  
본 연구는 기존의 RetinaFace 모델을 기반으로, 기존의 FPN 구조를 Bi-Directional Feature Pyramid Network(BiFPN)으로 대체하여 FPN의 한계를 보안한 새로운 모델인 BiRetina를 개발하였다.  
BiRetina는 기존 모델에서 사용되는 Feature Pyramid Network(FPN)의 단순한 피쳐맵(Feature Map) 합성 방법을 개선하기 위해 Bi-Directional Feature Pyramid Network(BiFPN)을 도입하고, 이를 핵심 구조로 삼는다.  
BiFPN은 각 피쳐맵에 대해 동적으로 가중치를 부여한 후 합성하는 방식을 채택함으로써, 피쳐(feature) 간의 중요도를 보다 정교하게 반영한다. 이러한 방식은 단순한 합성 방법에 비해 피쳐 정보의 손실을 줄이고, 세밀한 특징을 더욱 잘 포착할 수 있도록 한다.  
또한, 본 연구에서는 BiRetina의 구조를 최적화하여 레이어 수를 감소시키고, 활성화 함수로 ELU를 도입하며, AdamW 최적화 알고리즘을 적용함으로써 모델의 효율성과 성능을 동시에 개선한다. 특히, MobileNet을 backbone으로 사용한 실험에서 이러한 구조적 변경이 얼굴 감지 성능을 약  2% 향상시키는 결과를 보여준다.  






## 설치
#### clone
1. git clone https://github.com/dohun-mat/BiRetina

#### 데이터

1.  [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) 데이터셋을 다운로드

2. [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0) 에서 얼굴 경계 상자 및 5개의 얼굴 랜드마크를 다운로드

3. 데이터셋 디렉토리를 다음과 같이 구성합니다.

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wide_val.txt에는 val 파일 이름만 포함되고 레이블 정보는 포함되지 않습니다.

##### Data1
또한 위의 디렉터리 구조와 같이 사용된 정리된 데이터 세트를 제공합니다.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## 훈련
[google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)이 링크에서 MobilenetV1X0.25_pretrain.tar를 다운로드 받습니다.
```Shell
  ./weights/
      mobilenetV1X0.25_pretrain.tar
```
1. 훈련하기 전에 에서 네트워크 구성(예: 배치_크기, 최소 크기 및 단계 등)을 확인할 수 있습니다.  
   ```data/config.py and train.py```

3. WIDER FACE를 사용하여 모델을 훈련합니다.
  ```Shell
  python train.py --network mobile0.25
  ```


## 평가
1. txt 파일 생성
```Shell
python test_widerface.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25
```
2. txt 결과를 평가합니다. 데모는 [Here](https://github.com/wondervictor/WiderFace-Evaluation) 에서 제공됩니다. 
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth
```
3. [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) 에서 wideface 공식 Matlab 평가 데모를 사용할 수도 있습니다.  

## 기존 RetinaFace와 BiRetina모델의 이미지 감지 비교 사진(위 : Retinaface, 아래 : BiRetina)  

![image](https://github.com/dohun-mat/BiRetina/assets/81942144/5af2b544-9d08-476c-85a9-1c71550a2dac)



