# Deep Learning

---

### **Index:**
- [Good Lectures in Deep learning](#lecture)
- [Datasets for Deep Learning](#datasets)
- [Lab: Tools and Codes](#lab)
- [Keras](#keras)
- [Main Content](#content)
  1. [Introduction to Deep Learning](#introduction)
  2. [Regularization for Deep Learning](#regularization)
  3. [Optimization for Training Deep Models](#optimization)
  4. [CNN in Deep learning ](#cnn)
  5. [RNN in Deep learning](#rnn)
  6. [Autoencoder](#autoencoder)
  7. [Representation Learning](#representation)
  8. [GAN](#gan)
- [miscellany](#misc)
  - [History of Deep Learning](#history)
  - [Amount of Data for Deep Learning](#size_data)


---

### <a name="lecture"></a>Good Lectures in Deep learning
   * Book: [Deep Learning](https://www.deeplearningbook.org) by Ian Goodfellow and Yoshua Bengio and Aaron Courville
   * Lecture: [Deep Learning Course](https://cedar.buffalo.edu/~srihari/CSE676/index.html) Complete Lectures for Deep Learning Course by Sargur Srihari
   * Lecture: [Overview of ways to improve generalization](http://www.cs.toronto.edu/~hinton/coursera/lecture9/lec9.pdf) by Geoffrey Hinton, et al.


### <a name="datasets"></a>Datasets for Deep Learning
  * Blog: [Visual Data](https://www.visualdata.io)


### <a name="lab"></a> Lab: Tools and Codes
  * Blog: [Step-by-step Guide to Install TensorFlow 2](https://medium.com/@cran2367/install-and-setup-tensorflow-2-0-2c4914b9a265)
  * Blog: [Tensorflow Hub](https://www.tensorflow.org/hub) A library for the publication, discovery, and consumption of reusable parts of machine learning models
  * Blog: [TensorFlow Models](https://github.com/tensorflow/models/) A repository contains a number of different models implemented in TensorFlow
  * Software: [Learning rate multiplier wrapper for optimizers](https://pypi.org/project/keras-lr-multiplier/)
  
    ```
    Data Processing in Tensorflow
    ```
  * Blog: [Datasets for Estimators](https://www.tensorflow.org/guide/datasets_for_estimators)
  * Blog: [Feature Columns](https://www.tensorflow.org/guide/feature_columns)
    ```
    Decoration in Tensorflow
    ```
  * Blog: [tf.function](https://www.tensorflow.org/beta/tutorials/eager/tf_function) converts a python method to its computation graph in TF
  * Blog: [Analyzing tf.function to discover AutoGraph strengths and subtleties](https://pgaleone.eu/tensorflow/tf.function/2019/05/10/dissecting-tf-function-part-1/)
  [, part-2](https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/)
  [, part-3](https://pgaleone.eu/tensorflow/tf.function/2019/05/10/dissecting-tf-function-part-3/)
    ```
    TensorBoard
    ```
   * Blog: [Introduction to TensorBoard and TensorFlow visualization](https://adventuresinmachinelearning.com/introduction-to-tensorboard-and-tensorflow-visualization/)
   * Blog: [Introduction to TensorBoard](https://www.easy-tensorflow.com/tf-tutorials/basics/introduction-to-tensorboard)
   * Blog: [TensorBoard Tutorial](https://www.datacamp.com/community/tutorials/tensorboard-tutorial) 
     ```
     Keras
     ```    
   * Blog: [How to Load, Convert, and Save Images With the Keras API](https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/) by Jason Brownlee
   * Blog: [How to Load Large Datasets From Directories for Deep Learning in Keras](https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/) by by Jason Brownlee
   * Blog: [Tutorial on Keras ImageDataGenerator with flow_from_dataframe](https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1)
   * Blog: [How to Use The Pre-Trained VGG Model to Classify Objects in Photographs](https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/)
  
## <a name="content"></a>Main Content: Slides, lectures and academic descriptions of DL concepts

1.  ### <a name="introduction"></a>Introduction to Deep Learning
  * Blog: [Tensor](https://en.wikipedia.org/wiki/Tensor) A "tensor" is a multilinear function of several vector variables
  * Blog: [Quick ML Concepts: Tensors](https://towardsdatascience.com/quick-ml-concepts-tensors-eb1330d7760f)
  * Blog: [A Gentle Introduction to Tensors for Machine Learning with NumPy](https://machinelearningmastery.com/introduction-to-tensors-for-machine-learning/)
  * Blog: [Understanding Tensors and Graphs](https://www.analyticsvidhya.com/blog/2017/03/tensorflow-understanding-tensors-and-graphs/)

2.  ### <a name="regularization"></a>Regularization for Deep Learning  
```
Required Reading:
```
  * [Chapter 7](http://www.deeplearningbook.org/contents/regularization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Training Deep Neural Networks](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/slides/lec4-training-deep-nets.pdf) by Aykut Erdem
```
    Additional Reading:
```
  * [How to Improve Deep Learning Model Robustness by Adding Noise](https://machinelearningmastery.com/how-to-improve-deep-learning-model-robustness-by-adding-noise/) by Jason Brownlee 
    * Slide: [Regularization for Deep Learning](https://www.deeplearningbook.org/slides/07_regularization.pdf)  by Ian Goodfellow


3.  ### <a name="optimization"></a>Optimization for Training Deep Models  
    ```
      Required Reading:
    ```
  * [Chapter 8](http://www.deeplearningbook.org/contents/optimization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br> 
    * Slide: [Training Deep Neural Networks](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/slides/lec4-training-deep-nets.pdf) by Aykut Erdem
    * Slide: [Gradient Descent and Structure of Neural Network Cost Functions](https://www.deeplearningbook.org/slides/sgd_and_cost_structure.pdf) by Ian Goodfellow
    * Slide: [Tutorial on Optimization for Deep Networks](https://www.deeplearningbook.org/slides/dls_2016.pdf) by Ian Goodfellow    
    * Blog: [Neural Network Optimization](https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0) by Matthew Stewart  
    * Slide: [Batch Normalization in Deep Networks](https://www.learnopencv.com/batch-normalization-in-deep-networks/) by Sunita Nayak 
    * Blog: [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/) by Sebastian Ruder 
    ```
    Additional Reading:
    ```
   * [Video](https://www.youtube.com/watch?v=Xogn6veSyxA) of lecture / discussion: This video covers a presentation by Ian Goodfellow and group discussion on the end of Chapter 8 and entirety of Chapter 9 at a reading group in San Francisco organized by Taro-Shigenori Chiba. <br>
   * Slides: Optimization for Training Deep Models [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L15-opt.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L16-opt-2.pdf) by U Kang 
   * Blog: [Why Momentum Really Works](https://distill.pub/2017/momentum/) by Gabriel Goh  
   * Blog: [Preconditioning the Network](https://cnl.salk.edu/~schraudo/teach/NNcourse/precond.html) by Nic Schraudolph and Fred Cummins  
   * Blog: [How to Accelerate Learning of Deep Neural Networks With Batch Normalization](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/) by Jason Brownlee  
   * Slide: [Conjugate Gradient Descent](http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf) by Aarti Singh
   * Blog: [Orthogonal Matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix) in Wikipedia

4.  ### <a name="cnn"></a>CNN in Deep learning 
* Blog: [A Comprehensive Tutorial to learn Convolutional Neural Networks from Scratch](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/) by PULKIT SHARMA
* Blog: [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) by Chi-Feng Wang
* Blog: [Depth wise Separable Convolutional Neural Networks](https://www.geeksforgeeks.org/depth-wise-separable-convolutional-neural-networks/)

5.  ### <a name="rnn"></a>RNN in Deep learning 
  * Blog: [Gentle introduction to Echo State Networks](https://towardsdatascience.com/gentle-introduction-to-echo-state-networks-af99e5373c68)
  * Slide: [ECho State Network](http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/aa2/rnn4-esn.pdf)
  * Blog: [ECho State Network](http://www.scholarpedia.org/article/Echo_state_network) by Herbert Jaeger
  * Blog: [ECho State Network](https://en.wikipedia.org/wiki/Echo_state_network) in Wikipedia
  * Paper: [How to Construct Deep Recurrent Neural Networks](https://arxiv.org/abs/1312.6026)
  * Person: [Alex Graves](https://www.cs.toronto.edu/~graves/)
  * Blog: [Reservoir Computing](https://www.researchgate.net/post/what_is_the_realitionship_between_deep_learning_methods_and_reservoir_computing_if_any)
  * Code: [ESN](https://github.com/ciortanmadalina/EchoStateNetwork/blob/master/EchoStateNetwork.ipynb)

6.  ### <a name="autoencoder">Autoencoder
  * Blog: [Introduction to autoencoders](https://www.jeremyjordan.me/autoencoders/) by Jeremy Jordan
  * Paper: [What Regularized Auto-Encoders Learn from the Data Generating Distribution](https://arxiv.org/abs/1211.4246) by  Guillaume Alain, Yoshua Bengio
  
7.  ### <a name="representation">Representation Learning
  * Blog: [How neural networks learn distributed representations](https://www.oreilly.com/ideas/how-neural-networks-learn-distributed-representations)

8.  ### <a name="gan"></a>GAN
  * Video: [Adversarial Machine Learning](https://videos.videoken.com/index.php/videos/aaai-2019-videos-invited-talk-ian-goodfellow-google-ai-adversarial-machine-learning/) by Ian Goodfellow 

## <a name="misc"></a>Miscellaneous Matterials

### <a name="history"></a>History of Deep Learning  
  * Paper: [On the Origin of Deep Learning](https://arxiv.org/abs/1702.07800) by Haohan Wang and Bhiksha Raj
  
### <a name="size_data"></a>Amount of Data for Deep Learning  
* Blog: [How Do You Know You Have Enough Training Data?](https://towardsdatascience.com/how-do-you-know-you-have-enough-training-data-ad9b1fd679ee) by Theophano Mitsa
* Blog: [One in ten rule](https://en.wikipedia.org/wiki/One_in_ten_rule) in Wikipedia
* Paper: [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/pdf/1712.00409.pdf) by J. Hestness, et al.
* Presentation: [Scaling Deep Learning on Multi-GPU Servers](https://discan18.github.io/assets/presentations/peter.pdf)
