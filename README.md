# Image Stitching & Panoramic Blending

이 프로젝트는 3장 이상의 여러 이미지를 자연스럽게 하나의 큰 파노라마 이미지로 정합(Stitching)하는 컴퓨터 비전 과제입니다. 고전적인 Local Feature(BRISK) 방식뿐만 아니라 최첨단 딥러닝 방식(LoFTR)을 모두 구현하였으며, 파노라마 합성 시 흔히 발생하는 잔상(Ghosting)과 왜곡(Distortion)을 완벽하게 제거하기 위해 상용 소프트웨어 수준의 심화 최적화 기법을 적용했습니다.

## 🌟 핵심 기능 (Key Features)

### 1. Cylindrical Projection (원통형 투영)
* 카메라를 좌우로 회전(Panning)하며 촬영한 사진들을 평면(Planar)으로 단순 투영할 경우 가장자리가 심하게 늘어나는 원근 왜곡이 발생합니다.
* 이를 해결하기 위해 입력 이미지를 미리 원통형 좌표계로 역맵핑(Backward Mapping)하여 왜곡 없이 자연스럽게 3장 이상의 이미지를 계속해서 이어붙일 수 있는 파이프라인을 구축했습니다.

### 2. Deep Learning 기반 특징점 매칭 (LoFTR)
* `main_loftr.py` 스크립트에는 기존의 고전적인 방식(BRISK) 대신 Transformer 구조를 활용한 최첨단 매칭 모델인 **LoFTR**를 도입했습니다.
* 질감이 부족한 영역(하늘, 밋밋한 벽 등)이나 조명 변화가 있는 상황에서도 극강의 픽셀 단위 매칭 정확도를 보여줍니다.
* **메모리 최적화**: 고해상도 이미지 처리 시 발생하는 VRAM 메모리 초과(OOM)를 막기 위해, 매칭 연산은 저해상도 Tensor로 수행하고 추출된 특징점 좌표를 다시 원본 비율로 복원(Scale-back)하는 실용적인 추론 파이프라인을 독자적으로 설계했습니다.

### 3. Least Squares Refinement (최소제곱 보정)
* 기본 RANSAC 알고리즘은 오직 무작위로 뽑은 "단 4개의 점"을 맹신하여 변환 행렬을 추정하기 때문에 4개 점 외곽의 다른 영역들에서는 심한 비틀림과 엇나감이 발생합니다.
* 이를 해결하기 위해 RANSAC이 걸러낸 **수십~수백 개의 정상 매칭점(Inliers) 전체**를 한데 모아 SVD(특이값 분해) 연산에 다시 대입하여 최적의 변환 행렬(Homography)을 한 번 더 깎고 다듬는(Refinement) 고도화 과정을 구현했습니다.

### 4. Center-weighted Voronoi Seam Blending (심화 블렌딩)
* 두 이미지가 겹치는 구역(Overlap)을 단순 평균(Alpha Blending) 내어 넓게 섞게 되면, 1픽셀의 미세한 오차만 있어도 피사체가 두 개로 겹쳐 보이는 치명적인 잔상(Ghosting)이 남게 됩니다.
* 이를 원천 차단하기 위해, 두 이미지의 중심으로부터의 거리를 계산하여 정확히 거리가 일치하는 지점에 선명한 경계선(**Voronoi Seam**)을 찾았습니다. 
* 경계선의 좌/우측은 각각 단일 이미지만을 100% 사용하여 잔상 원인을 물리적으로 없앴고, 해당 경계선에만 약 30픽셀 너비의 얇은 Gaussian Blur 마스크를 씌워 색상의 이질감 없이 물감처럼 스며들듯 부드럽게 융합(Blending)되도록 처리했습니다.

### 5. Auto Inner-Bounding Box Crop (자동 크롭 로직)
* 정합이 완료된 파노라마 이미지는 카메라의 손떨림 궤적과 상하 끄덕임에 따라 가장자리에 지그재그 모양의 검은색 빈 공간(Padding)이 발생합니다.
* 이를 깔끔한 직사각형 액자 뷰로 잘라내기 위해, 이미지의 상/하/좌/우 가장자리 네 방향에서 검은 픽셀이 하나라도 검출되면 안쪽으로 1픽셀씩 깎아 들어가는 `Inner Crop` 알고리즘을 구현하여 완벽한 형태의 결과물만을 자동 출력합니다.

## 🚀 실행 방법 (How to Run)

### 필수 요구사항
```bash
pip install numpy opencv-python torch kornia
```

### 1. 기본 버전 (OpenCV BRISK 기반)
```bash
python main.py
```
* **특징**: CPU만으로도 매우 빠르게 동작하며, RANSAC Refinement와 Voronoi Blending의 힘으로 고전적 방식임에도 높은 퀄리티와 선명도를 보장합니다.

### 2. 딥러닝 버전 (PyTorch LoFTR 기반)
```bash
python main_loftr.py
```
* **특징**: 첫 실행 시 LoFTR 모델 가중치를 자동으로 다운로드합니다. Mac 환경에서는 `mps` 하드웨어 가속을 완벽 지원합니다. BRISK로 놓치는 픽셀 단위 디테일까지 강건하게 잡아내어 상용 스마트폰 파노라마 모드에 버금가는 극한의 선명도를 보여줍니다.

## 📁 파일 구조 (Directory Structure)
* `main.py` : BRISK 특징점 추출 기반의 파노라마 정합 파이프라인
* `main_loftr.py` : Kornia LoFTR 딥러닝 기반의 파노라마 정합 파이프라인
* `homography_estimation_implement.py` : 수제작 RANSAC 및 최적화 SVD 연산 로직
* `image_warping_implement.py` : Homography 행렬 기반 Vectorized 맵핑 로직
* `data/` : 원본 사진(`.JPG`) 및 생성된 파노라마 결과물, 단계별 캡처본이 저장되는 폴더
