# MMCaption

## Setup
```
conda create -n mm-mamba python=3.10 -y
conda activate mm-mamba

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

cd selective_scan
pip install .
```

## Prepare Data
1. Download coco annotations from [here](https://github.com/ttengwang/ClipCap/releases/download/v1.0/data.zip).
2. Put ```train_caption.json```, ```val_caption.json``` and ```test_caption.json``` under ```data/coco2014/annotations```
3. Download ```coco_karpathy_val_gt.json``` and ```coco_karpathy_test_gt.json``` from [here](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json) and [here](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json). Put them under ```data/coco2014/annotations```.
4. Download coco2014 images from coco website. Put them under ```data/coco2014/images```

Ensure your folder structure matches the following layout:

```
- data
  - coco2014
    - annotations
      - coco_karpathy_test_gt.json
      - coco_karpathy_val_gt.json
      - test_caption.json
      - train_caption.json
      - val_caption.json
    - images
      - test2014
      - train2014
      - val2014

```

