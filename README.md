# Semantic Image Segmentation with U-Net

A TensorFlow/Keras computer-vision project that implements a **U-Net from scratch** for multi-class semantic segmentation of road-scene images from the **CARLA self-driving simulation dataset**.

The goal is to predict a semantic class for every pixel in an RGB image. Instead of returning one label for the whole image, the network produces a dense segmentation mask that describes the scene at pixel level.

## Project overview

Semantic segmentation is a core perception task in autonomous-driving and robotics systems because the model needs to understand not only **what** is present in a scene, but also **where** each object or region is located.

This project covers the complete workflow:

- loading paired RGB images and semantic masks;
- building a TensorFlow `tf.data` preprocessing pipeline;
- resizing and normalizing the data;
- implementing reusable U-Net encoder and decoder blocks;
- adding skip connections between matching encoder and decoder stages;
- training a 23-class pixel-wise classifier;
- converting raw model logits into predicted masks;
- plotting the training curve; and
- visually comparing predictions with ground-truth segmentation masks.

## Model architecture

The model follows the classic U-Net encoder-decoder structure.

```text
Input: 96 x 128 x 3
        |
        v
Encoder Block 1: 32 filters
        |
        v
Encoder Block 2: 64 filters
        |
        v
Encoder Block 3: 128 filters
        |
        v
Encoder Block 4: 256 filters + Dropout(0.3)
        |
        v
Bottleneck: 512 filters + Dropout(0.3)
        |
        v
Decoder Block 1: 256 filters  <---- skip connection from Encoder 4
        |
        v
Decoder Block 2: 128 filters  <---- skip connection from Encoder 3
        |
        v
Decoder Block 3: 64 filters   <---- skip connection from Encoder 2
        |
        v
Decoder Block 4: 32 filters   <---- skip connection from Encoder 1
        |
        v
3 x 3 Conv2D: 32 filters
        |
        v
1 x 1 Conv2D: 23 output channels
        |
        v
Per-pixel class logits
```

The recorded model contains **8,640,471 trainable parameters**.

### Encoder

Each encoder block contains:

1. a `3 x 3` convolution;
2. ReLU activation;
3. a second `3 x 3` convolution;
4. ReLU activation;
5. optional dropout; and
6. optional `2 x 2` max pooling.

The convolutional layers use `same` padding and He-normal weight initialization.

The feature map before pooling is preserved as a **skip connection** so that high-resolution spatial information can be reused by the decoder.

### Bottleneck

The deepest block uses **512 filters** and a dropout rate of **0.3**. Max pooling is disabled at this stage because it represents the bottom of the U-Net.

### Decoder

Each decoder stage:

1. upsamples the previous feature map with `Conv2DTranspose`;
2. concatenates it with the corresponding encoder feature map;
3. applies two `3 x 3` convolution layers with ReLU activations.

These skip connections help the decoder recover local detail that would otherwise be lost during repeated downsampling.

### Output layer

A final `1 x 1` convolution maps the final decoder representation to **23 output channels**.

Each output channel represents the model's logit for one semantic class at every pixel.

## Dataset and preprocessing

The notebook expects paired PNG files in the following structure:

```text
data/
├── CameraRGB/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── CameraMask/
    ├── 000001.png
    ├── 000002.png
    └── ...
```

RGB images and masks are paired by filename. This avoids relying on arbitrary filesystem ordering.

The preprocessing pipeline performs the following operations:

- decodes RGB images as three-channel PNG files;
- converts RGB images to `float32`, automatically scaling pixel values to `[0, 1]`;
- decodes segmentation masks;
- reduces the mask representation to a single class-ID channel;
- resizes images to **96 x 128**;
- resizes masks to **96 x 128**; and
- constructs a TensorFlow dataset for model training.

Nearest-neighbor resizing is used for the masks so that discrete class IDs are not blended during interpolation.

In the recorded notebook run, the raw mask shape is **480 x 640 x 1** and the processed mask shape is **96 x 128 x 1**.

## Training configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Input shape | `96 x 128 x 3` |
| Number of classes | 23 |
| Optimizer | Adam |
| Loss | Sparse Categorical Cross-Entropy |
| `from_logits` | `True` |
| Metric | Pixel accuracy |
| Batch size | 32 |
| Shuffle buffer | 500 |
| Epochs | 40 |
| Trainable parameters | 8,640,471 |

### Why sparse categorical cross-entropy?

Each pixel in the segmentation mask stores a single integer class ID rather than a one-hot encoded class vector.

For that reason, the project uses:

```python
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

The final layer does not apply softmax explicitly. It produces raw logits, which the loss function handles internally.

## Results

The recorded 40-epoch training run produced:

| Metric | First epoch | Final epoch |
|---|---:|---:|
| Loss | 3.1109 | **0.0809** |
| Training pixel accuracy | 41.36% | **97.52%** |

The training curve shows a steady increase in pixel accuracy as the network learns the scene labels.

The notebook also visualizes multiple examples containing:

- the original RGB image;
- the ground-truth semantic mask; and
- the model's predicted mask.

### Important evaluation note

**97.52% is training pixel accuracy, not test accuracy.**

The recorded notebook run does not create a held-out validation or test split. The result therefore demonstrates how well the model fits the available training data, but it does not by itself measure performance on unseen scenes.

A stronger evaluation would add a dedicated validation/test split and report segmentation-specific metrics such as:

- mean Intersection over Union (mIoU);
- per-class IoU;
- Dice/F1 score;
- validation loss; and
- validation pixel accuracy.

## Prediction pipeline

The model returns a tensor containing 23 class scores for every pixel.

The predicted mask is created by selecting the class with the highest logit:

```python
predicted_class = tf.argmax(prediction, axis=-1)
```

This converts the model output into one class ID per pixel, producing a single semantic segmentation mask.

## Repository structure

A simple project layout is:

```text
.
├── UNet_Semantic_Segmentation_Project.ipynb
├── README.md
└── data/
    ├── CameraRGB/
    └── CameraMask/
```

The dataset is expected locally under `data/` and is not embedded in the notebook.

## Requirements

The notebook metadata uses **Python 3.10**.

Main dependencies:

```text
tensorflow
numpy
matplotlib
imageio
jupyter
```

Install them with:

```bash
pip install tensorflow numpy matplotlib imageio jupyter
```

A GPU-enabled TensorFlow environment is recommended for faster training, although the code itself does not depend on GPU-specific APIs.

## Running the project

1. Clone or download the repository.
2. Create the `data/CameraRGB/` and `data/CameraMask/` directories.
3. Place matching RGB images and segmentation-mask PNG files in those folders.
4. Install the Python dependencies.
5. Start Jupyter:

```bash
jupyter notebook
```

6. Open `UNet_Semantic_Segmentation_Project.ipynb`.
7. Run the notebook from top to bottom.

The notebook will:

- verify that the dataset directories exist;
- pair matching image and mask filenames;
- display sample data;
- build the preprocessing pipeline;
- construct the U-Net;
- display the model summary;
- train for 40 epochs;
- plot training accuracy; and
- display segmentation predictions.

## Implementation details

### Robust image/mask pairing

The project pairs files using their filenames rather than trusting the order returned by the operating system:

```python
paired_filenames = sorted(set(image_files) & set(mask_files))
```

This helps prevent an RGB frame from accidentally being trained against the wrong mask.

### Skip connections

The encoder returns both:

- a tensor passed to the next encoder stage; and
- a higher-resolution tensor saved for the decoder.

The decoder concatenates these saved tensors with its upsampled feature maps.

### Dropout

A dropout probability of `0.3` is used in the deeper encoder stages, where the network has the highest number of filters.

### Multi-class output

The model predicts 23 values for every output pixel. `argmax` converts those values into the final semantic class assignment.

## Current limitations

The current version is intentionally focused on implementing and understanding the U-Net pipeline. Its main limitations are:

- no train/validation/test split;
- no unseen-data benchmark;
- no mIoU or per-class IoU reporting;
- no data augmentation;
- no checkpointing or early stopping;
- no class-imbalance analysis; and
- no exported production inference model.

Because pixel accuracy is calculated across every pixel, a complete segmentation evaluation should include class-aware metrics such as mIoU in addition to accuracy.

## Possible improvements

Future versions could extend the project by:

- adding deterministic train/validation/test splits;
- using `tf.data` prefetching and parallel mapping;
- adding image augmentation;
- logging validation mIoU and per-class IoU;
- using `ModelCheckpoint` and `EarlyStopping`;
- adding a confusion matrix across semantic classes;
- comparing the custom U-Net with U-Net++, DeepLabV3+, or SegFormer;
- experimenting with pretrained encoder backbones;
- exporting the model as a SavedModel or TensorFlow Lite model; and
- building a small inference application for uploaded road-scene images.

## Skills demonstrated

This project demonstrates practical experience with:

- semantic image segmentation;
- convolutional neural networks;
- U-Net architecture;
- TensorFlow and Keras;
- `tf.data` pipelines;
- image and mask preprocessing;
- transposed convolutions;
- skip connections;
- dropout regularization;
- sparse categorical cross-entropy;
- model training and visualization; and
- qualitative computer-vision evaluation.

## Key takeaway

U-Net combines a compressed, high-level representation of an image with fine spatial information from earlier layers. The encoder learns **what** is present, while skip connections help the decoder recover **where** those features belong.

The result is a network capable of producing dense, pixel-level predictions while preserving significantly more spatial detail than a conventional classification CNN.
