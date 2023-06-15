## Installation

### Requirements:
- PyTorch 
- torchvision
- cocoapi
- yacs
- numpy
- matplotlib
- GCC
- OpenCV
- CUDA


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name sgg_attack python=3.7 -y
conda activate sgg_attack

# this installs the right pip and dependencies for the fresh python
conda install ipython h5py nltk joblib jupyter pandas scipy

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs>=0.1.8 cython matplotlib tqdm opencv-python numpy>=1.19.5

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge timm einops

# install pycocotools
conda install -c conda-forge pycocotools

# install cityscapesScripts
python -m pip install cityscapesscripts


git clone ~ # code for attack
cd sgg_attack
# install Scene Graph Detection
python setup.py build develop

```
