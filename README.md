
## 설치
#### clone
1. git clone https://github.com/dohun-mat/FeatherFace

## 데이터
정리된 데이터 세트를 제공합니다.
Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

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
   CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --network mobile0.25
  ```


## 평가
1. txt 파일 생성
```Shell
python test_widerface.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --origin_size True
```
2. txt 결과를 평가합니다. 데모는 [Here](https://github.com/wondervictor/WiderFace-Evaluation) 에서 제공됩니다. 
```Shell
cd ./widerface_evaluate
python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth
```
3. [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) 에서 wideface 공식 Matlab 평가 데모를 사용할 수도 있습니다.  


