# 3D-FISH-classifier-for-RI-tomography

Please cite the following paper when using this classifier code. 

Label-free bone marrow white blood cell classification using refractive index tomograms and deep learning
(https://www.biorxiv.org/content/10.1101/2020.11.13.381244v1)


Sample data can be found [here](https://drive.google.com/drive/folders/1DvD2xswLcMnz2Abn5tpnggzyMfux4Vuy?usp=sharing)
The full dataset can be shared by the authors with a reasonable request (donghun.ryu29@gmail.com or yk.park@kaist.ac.kr)


Train with a single gpu and batch_size (Also refer to main.py for various parsing options) 

```sh
python3 main.py --gpus 0 --batch_size 8 --save dir YOUR_SAVE_DIRECTORY
```

Activate virtual environment 

```sh
source pytorch/bin/activate
```
