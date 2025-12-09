# LLM_Tutorial

**Llama를 중심으로 한 LLM의 전반적 사용법을 담은 리포지토리**

## 포함 내용

* 가상 환경 세팅과 이를 위한 도커 파일
* 특정 목적을 위한 Llama 3.1 8B의 전반적인 파인튜닝 코드 (SFT + DPO)
* vLLM의 유동 배치를 활용한 빠른 추론 코드

> 모든 코드는 단일 GPU 환경에서 구동되도록 만들어졌으나, 아주 약간만 손보면 Multi-GPU에서의 가속화가 가능하도록 설계되었습니다.

## 파일 설명

* `Dockerfile`
> 도커 컨테이너 이미지를 빌드하기 위한 파일입니다. 호스트 서버에서 `sudo docker build –t {image-name} –f Dockerfile`로 빌드하세요.
* `Tutorial.ipynb`
> Llama 모델 테스트, SFT + DPO, vLLM 빌드까지의 과정을 글로 정리한 문서입니다. 참고용으로 사용해주세요. 실전 코드는 `main` 폴더에 들어있습니다.