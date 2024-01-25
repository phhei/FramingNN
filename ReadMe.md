# A cross-parallel-dataset approach to frame classification at variable granularity levels

With this code, you can reproduce the results of the paper
["_A cross-parallel-dataset approach to frame classification at variable granularity levels_"](https://www.degruyter.com/document/doi/10.1515/itit-2020-0054/html)
In this file, we explain how you can set up and use this application.

## Set up

We recommend to use Python **3.8**.

### Required Libraries

- Natural language toolkit: ``pip install nltk`` (we applied version 3.4.5)
  - we need certain [NLTK data](https://www.nltk.org/data.html). However, this is done in the code itself
- Tensorflow (the use of GPU is optional): ``pip install tensorflow`` (we applied version 2.3.0)
- for logging: ``pip install loguru`` (we applied version 0.4.1)
- Word-Movers-Distance-implementation: ``pip install word-mover-distance`` (we applied version 0.0.1)
- for the plots: ``pip install matplotlib`` (we applied version 3.2.2)

And of course basic libraries:

- ``pip install numpy`` (we applied version 1.19.1/ installed with tensorflow, if there are some CPU/GPU-errors try to ``pip install --upgrade tensorflow``)

### Datasets

We use two datasets

#### The Webis-Argument-Dataset

See [this homepage](https://webis.de/data/webis-argument-framing-19.html)

Yamen Ajjour, Milad Alshomary, Henning Wachsmuth, and Benno Stein. Modeling Frames in Argumentation. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP 2019), pages 2922-2932, November 2019. ACL.

#### The Media-Frames-Dataset

See [this GitHub-reference](https://github.com/dallascard/media_frames_corpus)

Card, Dallas, et al. "The media frames corpus: Annotations of frames across issues." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2015.

Unfortunately, we can't provide the dataset in this repository due to license issues.

### Important files for evaluation

In principle, there is a pre-defined step: the creation of the dataset and then the evaluation on it.
There are scripts to convert the raw datasets in a generalized format, marked with ``_out.csv``.
The preprocessor-scripts are in the folder _Corpora_. A very important script is the ``UserFrames2GenericFrames.py``

There are two main files for evaluation.

#### ``BiLSTMApproach.py``

This file is for the single setup. Please execute the file with ``python3 BiLSTMApproach.py``

#### ``BiLSTMApproach_MultiTask.py``

This file is for the cross-parallel-dataset setup. Please execute the file with ``python3 BiLSTMApproach_MultiTask.py``

#### ``TrainGenereicFrames_ALBERT2.py``

Of course, we tried to apply the modern pre-trained transformers, too.
We used the [ALBERT](https://huggingface.co/transformers/model_doc/albert.html) from the
_huggingface_-library to this end.
However, this does not lead to acceptable results. Hence, we do not consider this option further.

## Important parameters

Besides to the comment block at the beginning of the files we want to present the important parameters here.

### Predict the right output

We implemented several modes for predicting an output. We present the modes now.

#### Predict the embedded user label (token by token)

This mode is activated with ``using_word_embedding_output = True``. In this mode, we predict ``max_seq_len_output``
times ``embedding_size_output``d vectors which represents the tokenized frame embedded with pre-computed
word embeddings which are defined with ``word_embeddings_output``.

#### Predict the Frames-set-classes

To this end, ``frames`` is set to a ``GenericFrame``-instance in the ``Frames.py``. In this mode, we predict the
generic frame. However, if the label does not match exactly to a frame of the Frames-set, we discard the sample if
``filter_unknown_frames = True`` or label it with the ``unknown``-Frame. This mode is not recommended in combination
with the Webis-dataset.

#### Predict the mapped Frames-set-classes

To this end, ``frames=None`` and ``one_hot_output_clusters`` should be defined with an ``GenericFrame``-instance
in the ``Frames.py``.
The output-vector is determined by the [Word-movers-distance](https://pypi.org/project/word-mover-distance/).

#### Predict the right cluster ([k-means-algorithm by nltk](https://www.kite.com/python/docs/nltk.cluster.KMeansClusterer))

To this end, ``frames=None`` and ``one_hot_output_clusters`` should be defined with an integer representing the _k_.

The semantic clustering is activated by default.

### Further parameters for both settings (single and cross-parallel-dataset)

in cross-parallel-dataset setting, the following parameters are applied to each task input and output.

#### ``max_seq_len``

We described in our paper a fixed length for each input (premise+conclusion). Here we can define this length.
Inputs with a smaller length will be padded, inputs with a longer length will be discarded.

#### ``enable_fuzzy_framing``

This boolean flag enables with a ``True``-value the fuzzy framing which means the disabling of the one-hot-encoding.
For example, consider an input that belongs to 80% to the first frame class and to 20% the second one.

If ``enable_fuzzy_framing`` is:

- ``True``: we want to predict [0.8 0.2 ...]
- ``False``: we want to predict [1 0 ...]

#### ``using_topic``

A boolean Flag to either include the **topic** in the input to the learning model or not.

#### ``using_premise``

A boolean Flag to either include the **premise** in the input to the learning model or not.

#### ``using_conclusion``

A boolean flag to either include the **conclusion** in the input to the learning model or not.

#### ``filter_unknown_frames``

A boolean flag which can be activated for an post-filtering. Normally, we filter in the step of the dataset creation.
However, this variable acn be used to filter frames which are not occurring the defined generic frame-class-set.

#### ``word_embeddings`` + ``embedding_size``

Here is the possibility to define the used pre-computed word embeddings. Must be stored in a txt file.
The embedding size is a integer which represents the dimensionality of the word embeddings.

We recommend using the [GloVe-Word-Embeddings](https://nlp.stanford.edu/projects/glove/)

#### ``NN_which_used``

We offer three neural net architectures:

- ``BiLSTM``: a bidirectional neural net which has LSTMs as core layer
- ``BiGRU``: a bidirectional neural net which has GRUs as core layer
- ``CNN``: a convolutional neural net (without recurrent layers)

#### ``data_set``

The used dataset

### Further parameters for the cross-parallel-dataset setting

#### The other meaning of ``frames``

In the single-setting, ``frames`` activates the strict Frame-set-classes. However, in the Multi-Task-setting,
``frames`` defines the output classes for the Media-Frames-Task and does not influence the output for the Webis-task.

#### ``give_webis_extra_layers``

This boolean flag controls the architecture on the Webis-task-side. If ``True``, Webis gets some additional layers.
To be more specific, Webis gets

- an additional [Dropout-layer](https://keras.io/api/layers/regularization_layers/dropout) with a _rate of 0.66_
- an additional [Dense-layer](https://keras.io/api/layers/core_layers/dense)

#### ``soft_parameter_sharing_lambda``

This parameter expects an float in the range from 0 (exclusive) to 1 (inclusive).
It controls the parameter sharing mode:

- ``soft_parameter_sharing_lambda`` < 1: soft-parameter-sharing
- ``soft_parameter_sharing_lambda`` = 1: hard-parameter-sharing
