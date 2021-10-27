# Who's Waldo? Linking People Across Text and Images
This is the official repository of [Who's Waldo](https://arxiv.org/abs/2108.07253) (ICCV 2021). Below are some quick steps to get started.

## 1. Request access to the Who's Waldo [dataset](https://whoswaldo.github.io/dataset).
## 2. Create a new conda environment
```
conda create --name whos-waldo
conda activate whos-waldo
pip install -r requirements.txt
```
## 3. Data preprocessing
Run the following preprocessing scripts in the environment created above.
First generate annotations:
```
python preprocess/generate_annotations.py --output {annotation-output-dir}
```
Process textual information for each split:
```
python preprocess/create_txtdb.py --ann {annotation-output-dir} --output {txtdb-name} --split {split}
```
Process visual information for each split:
```
python preprocess/create_imgdb.py --output {imgdb-name} --split {split}
```
Note that you will need to extract features for the images before creating the imgdb. We recommend this [repo](https://github.com/peteanderson80/bottom-up-attention) for feature extraction.

## 4. Set up Docker container
run ```launch_container.sh``` with the appropriate paths for each argument. 

## 5. Training
Create a training config file as ```config/train-whos-waldo.json```
Inside the container, run 
```
python train.py --config {path to training config}
```

## 6. Inference (evaluation and visualizations)
Inside the container, run 
```
python infer.py
```
with the appropriate arguments which can be found in ```infer.py```.
