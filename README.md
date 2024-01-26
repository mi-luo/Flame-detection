
# Flame-detection(STFDN)
STFDN stands for Flame Detection Network with Saliency Prior Guidance and Task Information Decoupling , which is a network specifically designed to detect flames in images captured by unmanned aerial vehicles (UAVs) in forest environments.


## install

git clone https://github.com/mi-luo/Flame-detection  # clone  
cd Flame-detection  
pip install -r requirements.txt  #  install  



## Pretrained Checkpoints：

1. our model weights in STFDN：

/STFDN/STDFDN_pth/weights/best.pt

or

2. Download our model weights on Baidu Cloud Drive: https://pan.baidu.com/s/1XjHGzPQUtVLDFCubTeffbw?pwd=6w4b   





## Dataset of HRSID-FD：

1. Download our val datasets on Baidu Cloud Drive: https://pan.baidu.com/s/1pq7jGjF3RZAL6b0o-ide_w?pwd=smxq  

2. Download our publicly available dataset (HRSID-FD) on Baidu Cloud Drive：https://pan.baidu.com/s/1fZ0japGjMxOx5hgJQNuIbA?pwd=22fe  




## Train

GPU：
```python 
python train.py \
    ${--weights ${initial weights path}} \
    ${--cfg ${model.yaml path}} \
    ${--data ${dataset.yaml path}} \
    --device 0
```


## Val

GPU：
```python 
python val.py
    ${--data ${dataset.yaml path}} \
    ${--weights ${model.pt path(s)}} \
    --device 0
```


## Acknowledgements
  Our dataset is created based on the datasets provided in "Wildland Fire Detection and Monitoring Using a Drone-Collected RGB/IR Image Dataset" and "Aerial imagery pile burn detection using deep learning: The FLAME dataset". we would like to express our sincere gratitude to the organizations involved for their significant contributions in forest fire monitoring. Below are the links to their respective research papers:

  "Wildland Fire Detection and Monitoring Using a Drone-Collected RGB/IR Image Dataset": https://ieeexplore.ieee.org/document/9953997
  "Aerial imagery pile burn detection using deep learning: The FLAME dataset": https://www.sciencedirect.com/science/article/pii/S1389128621001201
  We sincerely appreciate the spirit of open sharing demonstrated by these organizations and hope that this dataset, built upon their datasets, can provide valuable resources for research and development in the field of forest fire protection.

*NOTE：Please note that the listing above is not in any particular order*



## Contact
For STFDN bug reports and feature requests, and feel free to ask questions and engage in discussions!
