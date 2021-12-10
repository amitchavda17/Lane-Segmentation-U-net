# LANE-SEGMENTATION USING U-NET

This repositroy contains a keras implementation of U-net model for segmenting images.The model is trained on a subset of BDDK Dataset

1) **Setup**
    Create a virtual enviornment and insall requirements
    ```
    pip install -r requirements.txt
    ```
2) Download weights:
   Download the pretrained model weights [click here](https://drive.google.com/drive/folders/193rgTa-5S0Yy6wTO18d3h9ZlkKgvX4aj?usp=sharing)
3) Training the model :
   ```lane_segmentation_unet.ipynb``` contains the  code for data processing,model creation and training.

4) Inference:
   ```
   python inference.py --source[path to image/image/folder] -- weights [path to weight file] --output  [path for storing results(img segmentation)] (default './results')
   ```

   