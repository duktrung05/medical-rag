---
dataset_info:
- config_name: all
  features:
  - name: question_idx
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: context
    dtype: string
  - name: title
    dtype: string
  - name: keyword
    dtype: string
  - name: topic
    dtype:
      class_label:
        names:
          '0': body-part
          '1': disease
          '2': drug
          '3': medicine
  - name: article_url
    dtype: string
  - name: author
    dtype: string
  - name: author_url
    dtype: string
  splits:
  - name: train
    num_bytes: 47451105.527858645
    num_examples: 39881
  - name: test
    num_bytes: 2637825.0534154763
    num_examples: 2217
  - name: validation
    num_bytes: 2635445.4187258817
    num_examples: 2215
  download_size: 23042113
  dataset_size: 52724376.0
- config_name: body-part
  features:
  - name: question_idx
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: context
    dtype: string
  - name: title
    dtype: string
  - name: keyword
    dtype: string
  - name: topic
    dtype:
      class_label:
        names:
          '0': body-part
          '1': disease
          '2': drug
          '3': medicine
  - name: article_url
    dtype: string
  - name: author
    dtype: string
  - name: author_url
    dtype: string
  splits:
  - name: train
    num_bytes: 5865381.0
    num_examples: 4473
  - name: test
    num_bytes: 326510.14285714284
    num_examples: 249
  - name: validation
    num_bytes: 325198.85714285716
    num_examples: 248
  download_size: 2610513
  dataset_size: 6517090.0
- config_name: disease
  features:
  - name: question_idx
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: context
    dtype: string
  - name: title
    dtype: string
  - name: keyword
    dtype: string
  - name: topic
    dtype:
      class_label:
        names:
          '0': body-part
          '1': disease
          '2': drug
          '3': medicine
  - name: article_url
    dtype: string
  - name: author
    dtype: string
  - name: author_url
    dtype: string
  splits:
  - name: train
    num_bytes: 19619623.8
    num_examples: 14121
  - name: test
    num_bytes: 1090673.7966857872
    num_examples: 785
  - name: validation
    num_bytes: 1089284.4033142128
    num_examples: 784
  download_size: 9663817
  dataset_size: 21799582.000000004
- config_name: drug
  features:
  - name: question_idx
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: context
    dtype: string
  - name: title
    dtype: string
  - name: keyword
    dtype: string
  - name: topic
    dtype:
      class_label:
        names:
          '0': body-part
          '1': disease
          '2': drug
          '3': medicine
  - name: article_url
    dtype: string
  - name: author
    dtype: string
  - name: author_url
    dtype: string
  splits:
  - name: train
    num_bytes: 8465180.4
    num_examples: 8802
  - name: test
    num_bytes: 470287.8
    num_examples: 489
  - name: validation
    num_bytes: 470287.8
    num_examples: 489
  download_size: 3785181
  dataset_size: 9405756.000000002
- config_name: medicine
  features:
  - name: question_idx
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: context
    dtype: string
  - name: title
    dtype: string
  - name: keyword
    dtype: string
  - name: topic
    dtype:
      class_label:
        names:
          '0': body-part
          '1': disease
          '2': drug
          '3': medicine
  - name: article_url
    dtype: string
  - name: author
    dtype: string
  - name: author_url
    dtype: string
  splits:
  - name: train
    num_bytes: 13500996.235853817
    num_examples: 12485
  - name: test
    num_bytes: 750475.8820730916
    num_examples: 694
  - name: validation
    num_bytes: 750475.8820730916
    num_examples: 694
  download_size: 6953819
  dataset_size: 15001948.0
configs:
- config_name: all
  data_files:
  - split: train
    path: all/train-*
  - split: test
    path: all/test-*
  - split: validation
    path: all/validation-*
  default: true
- config_name: body-part
  data_files:
  - split: train
    path: body-part/train-*
  - split: test
    path: body-part/test-*
  - split: validation
    path: body-part/validation-*
- config_name: disease
  data_files:
  - split: train
    path: disease/train-*
  - split: test
    path: disease/test-*
  - split: validation
    path: disease/validation-*
- config_name: drug
  data_files:
  - split: train
    path: drug/train-*
  - split: test
    path: drug/test-*
  - split: validation
    path: drug/validation-*
- config_name: medicine
  data_files:
  - split: train
    path: medicine/train-*
  - split: test
    path: medicine/test-*
  - split: validation
    path: medicine/validation-*
---
