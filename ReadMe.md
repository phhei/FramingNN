# A cross-parallel-dataset approach to frame classification at variable granularity levels

With this code, you can reproduce the results of the paper
["_A cross-parallel-dataset approach to frame classification at variable granularity levels_"](https://www.degruyter.com/document/doi/10.1515/itit-2020-0054/html)
In this file, we explain how you can set up and use this application.

## Set up

We recommend to use Python **3.8**.

### Required Libraries

See the file ``requirements.txt`` for the required libraries.

### Datasets

We use two datasets

#### The Webis-Argument-Dataset

See [this homepage](https://webis.de/data/webis-argument-framing-19.html)

Yamen Ajjour, Milad Alshomary, Henning Wachsmuth, and Benno Stein. Modeling Frames in Argumentation. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP 2019), pages 2922-2932, November 2019. ACL.

#### The Media-Frames-Dataset

See [this GitHub-reference](https://github.com/dallascard/media_frames_corpus)

Card, Dallas, et al. "The media frames corpus: Annotations of frames across issues." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2015.

Unfortunately, we can't provide the dataset in this repository due to license issues.

### Important files for dataset preperation

In principle, there is a pre-defined step: the creation of the dataset and then the evaluation on it.
There are scripts to convert the raw datasets in a generalized format, marked with ``_out.csv``.
The preprocessor-scripts are in the folder _Corpora_. A crucial script is the ``UserFrames2GenericFrames.py``

## Main.py

The main file is ``Main.py``. It is the entry point for the application. It is a command line application. You can call it with the following parameters (only the most important ones are listed here):

````text
  -r, --runs INTEGER RANGE        How many runs do you want to perform? (in
                                  each run, the model parameters are
                                  initialized randomly)  [default: 1;
                                  1<=x<=100]
  -train, --train_data_path FILE  You can specify the path to the training
                                  data here (should be preprocessed). Define
                                  multiple paths (by setting several -train
                                  <value>) in case of multi-task learning
  -trainfrac, --train_data_frac <FLOAT FLOAT>...
                                  You can specify the fraction to use as the
                                  training data here (between 0 and 1 as
                                  tuple). Define multiple paths (by setting
                                  several -train <value> <value>) in case of
                                  multi-task learning
  -dev, -val, --dev_data_path FILE
                                  You can specify the path to the development
                                  data here (should be preprocessed). Define
                                  multiple paths (by setting several -dev
                                  <value> <value>) in case of multi-task
                                  learning
  -devfrac, -valfrac, --dev_data_frac <FLOAT FLOAT>...
                                  You can specify the fraction to use as the
                                  development data here (between 0 and 1 as
                                  tuple). Define multiple paths (by setting
                                  several -devfract <value> <value>) in case
                                  of multi-task learning
  -test, --test_data_path FILE    You can specify the path to the testing data
                                  here (should be preprocessed). Define
                                  multiple paths (by setting several -test
                                  <value>) in case of multi-task learning
  -testfrac, --test_data_frac <FLOAT FLOAT>...
                                  You can specify the fraction to use as the
                                  testing data here (between 0 and 1 as
                                  tuple). Define multiple paths (by setting
                                  several -testfrac <value> <value>) in case
                                  of multi-task learning
  -in, --fct_input_process [w2v|general_w2v|rnn_w2v|rnn|llm|transformer]
                                  You can specify the method to preprocess the
                                  text here. Define the same amount of
                                  multiple methods (by setting several -in
                                  <value>) in case of multi-task learning
  -out, -target, --fct_output_process [categorical|categorical_most_frequent|categorical_all_mf_wo_other|categorical_all|cluster|cluster_3|cluster_5|cluster_10|cluster_15|cluster_25|cluster_100]
                                  You can specify the method to preprocess the
                                  target (class) here. Define the same amount
                                  of multiple methods (by setting several -out
                                  <value>) in case of multi-task learning
  -topics, --process_topics       Should the topic be included as well in the
                                  input to the model?
  -length, -len, -l, --max_length INTEGER
                                  Do you want to limit the maximum sequence
                                  length? (e.g., for RNNs) If yes, set a token
                                  limit (int) here
  -m, --model [rnn|gru|lstm|transformer]
                                  You can specify the model (type) to use here
                                  [default: rnn]
  -mp, --model_params <TEXT TEXT>...
                                  You can specify the model parameters (as a
                                  dictionary) here
  -lr, --learning_rate FLOAT RANGE
                                  Learning rate for training  [default: 0.001;
                                  1e-10<=x<=1.0]
  -hps, --hard_parameter_sharing  (only relevant for multi-task learning)
                                  Should the models share their parameters
                                  hardly? (the core components are the same)
  -sps, --soft_parameter_sharing FLOAT RANGE
                                  (only relevant for multi-task learning)
                                  Should the models share their parameters
                                  softly? (the core components are the same
                                  type and architecture, but different
                                  weights)  [0.001<=x<=2.0]
  -es, --early_stopping           Should the training be stopped early if the
                                  validation loss does not decrease anymore?

````

You can also call ``Main.py --help`` to get a list of all parameters and options.