## **Outline:**
- [Graph Nets Library](#graphnets)
  1. [Installation of Graph Nets Library via pip](#graphnets_pip)
  2. [Installation of Graph Nets Library via anaconda](#graphnets_conda)
  3. [Module Reference of Graph Nets](#graphnets_doc)
- [Deep Graph Library](gnn_libraries.md#dgl)

# <a href="graphnets">Install Graph Nets Library</a>

[Graph Nets](https://github.com/deepmind/graph_nets) is DeepMind's library for constructing graph networks in Tensorflow and Sonnet. 

There are some conflicts and errors in the installation of the graph nets library. In the following lines, you can find an easiest way to install this library with its dependent packages. 

## <a href='graphnets_pip'>Installation of Graph Nets Library via pip</a>    
The Graph Nets library can be installed from [pip](https://github.com/deepmind/graph_nets/#Installation).  

To install the Graph Nets library for CPU, run:  

1. $ pip install python==3.6 graph_nets "tensorflow>=1.15,<2" tensorflow-probability==0.8.0  

To install the Graph Nets library for GPU, run:

2. $ pip install python==3.6 graph_nets "tensorflow_gpu>=1.15,<2" tensorflow-probability==0.8.0  

## <a href='graphnets_conda'>Installation of Graph Nets Library via anaconda</a>

First, it is better to create a new environment, e.g. GNN:  

1. $ conda create --name GNN 

Then, activate your environment:  

2. $ conda activate GNN  

Next, install the requirement packages using conda:  

3. $ conda install python=3.6 tensorflow=1.15 tensorflow-probability=0.8.0 jupyterlab matplotlib  

Next, we need to install  dm-sonnet and graph_nets  packages using pip:  

4. $ pip install dm-sonnet graph_nets   

Finally, assign a name to your kernel (e.g. GNN):  

5. $ python -m ipykernel install --user --name GNN --display-name "GNN"    

## <a href='graphnets_doc'>Module Reference of Graph Nets </a>
[Documentation of Graph Nets](https://github.com/deepmind/graph_nets/blob/master/docs/graph_nets.md)

# <a href="gnn_libraries.md#dgl">Deep Graph Library </a>
[Deep Graph Library](https://www.dgl.ai) supports PyTorch and MxNet DL frameworks as backends.

For installation, visit [Install DGL page](https://docs.dgl.ai/install/index.html)

## <a href='dgl_pytorch'>Installation of DGL with pytorch as backend </a>

1. First, try to install a new environment in your anaconda:

$conda create --name DGL #(DGL here is your environment's name)

2. Activate your environment:
$conda activate

3. you must install python 3. 7:

$conda install -c anaconda python=3.7

4. install pytorch as backend:

$conda install -c pytorch pytorch

5. Now you can install dgl library easily:

$conda install -c dglteam dgl

## <a href='dgl_tf'>Installation of DGL with Tensorflow as backend </a>

1. First, try to install a new environment in your anaconda:

$conda create --name DGLTF #(DGLTF here is your environment's name)

2. Activate your environment
$conda activate

3.Install python version 3. 7:

$conda install -c anaconda python=3.7

4. Installation of Tensorflow as backend:
$ conda install tensorflow

5. Install tfdlpack package:
If you are using cpu version of tensorflow:

$pip install tfdlpack  

or
if you want to have a gpu version of tensorflow:

$pip install tfdlpack-gpu  # when using tensorflow gpu version
$export TF_FORCE_GPU_ALLOW_GROWTH=true # and add this to your .bashrc/.zshrc file if needed

6. Now you can install dgl library easily:

$conda install -c dglteam dgl




