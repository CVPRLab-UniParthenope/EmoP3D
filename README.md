# Spatio-Temporal Pyramidal Deep Learning in Emotion Recognition 

EmoP3D is a pyramidal neural network for emotion recognition, based on biological pyramidal neurons developed in [[1]](#1);

## Datasets & Results

Code is tested on eNTERFACE '05 [[2]](#2) and ICT Youtube [[3]](#3) datasets;

*   Faces are extracted using [FaceNet](https://github.com/davidsandberg/facenet).
*   Number of feature maps used is 3;
*   Samples have 16 frame depth with an overlap of
    8 frames i.e 16x100x100x1 depth, height, width, channels dimensions;
*   10-Fold Cross-Validation;
*   Mini-batch gradient descent with Momentum
    optimization:
*   Batch size of 100 samples;
*   Momentum decay: 0.9;
*   Learning rate of 0.00015 and exponential decay ratio of 0.1 every 15 epochs;
*   Leaky ReLU activation function;


### Results

|      Model     | eNTERFACE | ICT Youtube |
|:--------------:|:---------:|:-----------:|
| AVER-Geometric [[4]](#4) |   41.59%  |      -      |
|      KCMFA [[5]](#5)     |    58%    |      -      |
|    AVER-CNN [[4]](#4)    |    62%    |      -      |
| LSTM(A) Binary [[6]](#6) |     -     |    52.3%    |
|      MARN [[7]](#7)     |     -     |    54.2%    |
|  **EmoP3D (Ours)** |   71.47%  |     75%     |

## Usage

Code uses *Sparse Softmax Cross Entropy* as loss function, it doesn't need *One Hot Encoding*.

### Dependencies

*   Tensorflow 1.2+;
*   TQDM;
*   Numpy;
*   Packaging;

### FLAGS

#### Checkpoint and evaluation

|      Option     |   Type  |     Default    |                       Description                      |
|:---------------:|:-------:|:--------------:|:-------------------------------------------------------|
|  evaluate_every |  float  |        1       | Number of epoch for each evaluation (decimals allowed) |
| test_milestones |   list  |    15,25,50    | Each epoch where performs test                         |
| save_checkpoint | boolean |      False     | Flag to save checkpoint or not                         |
| checkpoint_name |  string | 3dpyranet.ckpt | Name of checkpoint file                                   |


#### Input

|       Option      |  Type  | Default |            Description           |
|:-----------------:|:------:|:-------:|:---------------------------------|
|     train_path    | string |    //   | Path to npy training set         |
| train_labels_path | string |    //   | Path to npy training set labels  |
|      val_path     | string |    //   | Path to npy val/test set         |
|  val_labels_path  | string |    //   | Path to npy val/test set labels  |
|     save_path     | string |    //   | Path where to save network model |


#### Input parameters

|    Option    | Type | Default |          Description          |
|:------------:|:----:|:-------:|:------------------------------|
|  batch_size  |  int |   100   | Batch size                    |
| depth_frames |  int |    16   | Number of consecutive samples |
|    height    |  int |   100   | Sample height                 |
|     width    |  int |   100   | Sample width                  |
|  in_channels |  int |    1    | Sample channels               |
|  num_classes |  int |    6    | Number of classes             |


#### Hyper-parameters settings

|     Option    |  Type | Default |                                  Description                                 |
|:-------------:|:-----:|:-------:|:-----------------------------------------------------------------------------|
|  feature_maps |  int  |    3    | Number of maps to use (strict model shares the number of maps in each layer) |
| learning_rate | float | 0.00015 | Learning rate                                                                |
|  decay_steps  |  int  |    15   | Number of epoch for each decay                                               |
|   decay_rate  | float |   0.1   | Learning rate decay                                                          |
|   max_steps   |  int  |    50   | Maximum number of epoch to perform                                           |
|  weight_decay | float |   None  | L2 regularization lambda                                                     |


#### Optimization 

|    Option    |   Type  |  Default |                              Description                              |
|:------------:|:-------:|:--------:|:----------------------------------------------------------------------|
|   optimizer  |  string | MOMENTUM | Optimization algorthim (GD - MOMENTUM - ADAM)                         |
| use_nesterov | boolean |   False  | Flag to use Nesterov Momentum (it works only with MOMENTUM optimizer) |                                            |


## References

<a name="1">[1]</a> Ullah, Ihsan, and Alfredo Petrosino. "Spatiotemporal features learning with 3DPyraNet." International Conference on Advanced Concepts for Intelligent Vision Systems. Springer, Cham, 2016.

<a name="2">[2]</a> Martin, Olivier, et al. "The enterface’05 audio-visual emotion database." Data Engineering Workshops, 2006. Proceedings. 22nd International Conference on. IEEE, 2006.

<a name="3">[3]</a> Morency, Louis-Philippe, Rada Mihalcea, and Payal Doshi. "Towards multimodal sentiment analysis: Harvesting opinions from the web." Proceedings of the 13th international conference on multimodal interfaces. ACM, 2011.

<a name="4">[4]</a> Noroozi, F., Marjanovic, M., Njegus, A., Escalera, S., and Anbarjafari, G. (2017). Audio-visual emotion recognition in video clips. IEEE Transactions on Affective Computing.

<a name="5">[5]</a> Wang, Y., Guan, L., and Venetsanopoulos, A. N. (2012). Kernel cross-modal factor analysis for information fusion with application to bimodal emotion recognition. IEEE Transactions on Multimedia, 14(3):597–607

<a name="6">[6]</a> Chen, M., Wang, S., Liang, P. P., Baltrušaitis, T., Zadeh, A., and Morency, L.-P. (2017). Multimodal sentiment analysis with word-level fusion and reinforcement learning. 

<a name="7">[7]</a> Zadeh, A., Liang, P. P., Poria, S., Vij, P., Cambria, E., and Morency, L.-P. (2018). Multiattention recurrent network for human communication comprehension. arXiv preprint arXiv:1802.00923.

