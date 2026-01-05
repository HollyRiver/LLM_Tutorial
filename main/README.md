# HuggingFace LLM과 공개 데이터셋을 활용한 RLHF

## 구성

|파일명|설명|
|:-:|:-:|
|`csv_to_json_dataset.py`|pandas.DataFrame 형태의 tabular dataset을 SFT/DPO/Inference 용도로 변환. 적당한 `json` 파일이 존재할 경우 해당 과정 생략 가능. 자세한 설명은 해당 파일의 상단 주석 참조|
|`SFT.py`|**Supervised Fine Tuning**을 수행하기 위한 코드. `yaml` 파일로 하이퍼파라미터를 파싱|
|`sft_generate.py`||