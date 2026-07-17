# Human Image Segmentation with PyTorch and U-Net 
 
## Overview

This project implements a deep learning pipeline for binary human image segmentation using PyTorch. The objective is to train a convolutional neural network that receives an RGB image as input and predicts a pixel-level foreground mask identifying the human region in the image.

The project covers the full segmentation lifecycle: loading an image-mask dataset, preparing synchronized augmentations, defining a custom PyTorch dataset, batching samples with `DataLoader`, building a pretrained U-Net model, training with segmentation-specific losses, saving the best checkpoint, and visualizing model predictions on validation samples.

The task is a binary semantic segmentation problem. Each input image has a corresponding ground-truth mask where foreground pixels represent the human subject and background pixels represent the non-human region. The model predicts a single-channel logit map, which is converted into a binary mask during inference.

## Objectives

- Understand the structure of an image-mask segmentation dataset.
- Use Albumentations to apply spatial augmentations consistently to images and masks.
- Build a custom PyTorch `Dataset` class for paired image-mask loading.
- Load segmentation data into batches with PyTorch `DataLoader`.
- Use a pretrained convolutional segmentation model for binary mask prediction.
- Create training and validation loops for a segmentation task.
- Save the best-performing model checkpoint based on validation loss.
- Perform inference by converting raw model logits into thresholded binary masks.

## Technical Stack

- **Python**: Core programming language used in the notebook.
- **PyTorch**: Tensor operations, model definition, training loop, backpropagation, optimization, and checkpointing.
- **segmentation-models-pytorch**: Provides the U-Net architecture and pretrained encoder integration.
- **timm EfficientNet-B0 encoder**: Pretrained encoder backbone used inside the U-Net model.
- **Albumentations**: Image-mask augmentation library used for segmentation-safe transformations.
- **OpenCV**: Image and mask reading, color conversion, and resizing.
- **NumPy**: Array manipulation and mask preprocessing.
- **Pandas**: CSV loading and dataset path management.
- **scikit-learn**: Train-validation splitting with `train_test_split`.
- **Matplotlib**: Visualization of images, ground-truth masks, and predicted masks.
- **tqdm**: Progress bars for training and validation loops.

## Dataset Description

The notebook uses the `Human-Segmentation-Dataset-master` dataset cloned from `parth1620/Human-Segmentation-Dataset-master`. The dataset metadata is read from `train.csv`, which contains two path columns:

- `images`: Path to the RGB training image.
- `masks`: Path to the corresponding ground-truth segmentation mask.

The notebook reads the CSV into a Pandas DataFrame and performs an 80/20 train-validation split:

- Total samples inferred from notebook output: `290`
- Training samples: `232`
- Validation samples: `58`
- Split function: `train_test_split(df, test_size=0.2, random_state=42)`

Each sample consists of an image and its aligned binary mask. Images are loaded as three-channel RGB inputs, while masks are loaded as grayscale arrays and converted into single-channel binary tensors.

## Project Workflow

1. **Environment and dependency setup**
   - Installs `segmentation-models-pytorch`, `albumentations`, and `opencv-contrib-python`.
   - Uses a CUDA device for GPU-accelerated model training.

2. **Dataset acquisition and inspection**
   - Clones the human segmentation dataset.
   - Loads `train.csv` into a DataFrame.
   - Reads a sample image-mask pair and visualizes the RGB image beside the ground-truth mask.

3. **Configuration definition**
   - Defines dataset paths, device, image size, batch size, learning rate, epoch count, encoder name, and pretrained weights.
   - Uses `IMG_SIZE = 320`, `BATCH_SIZE = 16`, `EPOCHS = 25`, and `LR = 0.003`.

4. **Train-validation split**
   - Splits the dataset into training and validation subsets using a fixed random seed.
   - Produces `232` training samples and `58` validation samples.

5. **Segmentation augmentation setup**
   - Defines separate augmentation pipelines for training and validation.
   - Applies spatial transforms to images and masks together so that mask alignment is preserved.

6. **Custom dataset construction**
   - Implements a `SegmentationDataset` class that loads image-mask pairs, preprocesses them, applies augmentations, converts them to tensors, and returns them to the training loop.

7. **Batch loading**
   - Wraps the training and validation datasets in PyTorch `DataLoader` objects.
   - Produces `15` training batches and `4` validation batches with batch size `16`.

8. **Model creation**
   - Builds a U-Net segmentation model with a pretrained `timm-efficientnet-b0` encoder.
   - Configures the network for binary segmentation with one output channel.

9. **Training and validation**
   - Trains the model for `25` epochs.
   - Computes a combined Dice and binary cross-entropy loss.
   - Evaluates validation loss after each epoch.
   - Saves the best model checkpoint whenever validation loss improves.

10. **Inference and visualization**
    - Loads the saved best checkpoint.
    - Runs inference on selected validation samples.
    - Applies sigmoid activation and a `0.5` threshold to produce binary masks.
    - Displays the original image, ground-truth mask, and predicted mask.

## Data Preprocessing and Augmentation

The notebook preprocesses every image-mask pair before it is returned by the dataset class. Images are read using OpenCV, converted from BGR to RGB, resized to `320 x 320`, transposed from height-width-channel format to channel-height-width format, cast to `float32`, and normalized by dividing pixel values by `255.0`.

Masks are read in grayscale mode, resized to the same spatial resolution, expanded to include a channel dimension, transposed to channel-first format, cast to `float32`, scaled using `255.0` to normalize values, and rounded to obtain binary mask values. This produces masks compatible with binary segmentation loss functions.

Albumentations is used because segmentation tasks require image and mask transforms to remain spatially synchronized. The notebook defines two augmentation pipelines:

- Training augmentations:
  - `Resize(320, 320)`
  - `HorizontalFlip(p=0.5)`
  - `VerticalFlip(p=0.5)`
- Validation augmentations:
  - `Resize(320, 320)`

The training augmentations increase variation in object orientation while preserving the pixel-level relationship between each image and its mask. The validation pipeline keeps preprocessing deterministic by resizing only.

## Custom PyTorch Dataset

The `SegmentationDataset` class extends PyTorch's `Dataset` abstraction and provides the project-specific logic required for paired segmentation data.

The dataset stores the DataFrame and augmentation pipeline during initialization. Its `__len__` method returns the number of rows in the DataFrame, and its `__getitem__` method retrieves one image-mask pair by index.

For each sample, the dataset:

- Reads the image path and mask path from the DataFrame row.
- Loads the image with OpenCV and converts it to RGB.
- Loads the mask as a grayscale image.
- Resizes both image and mask to `320 x 320`.
- Applies Albumentations transforms to the image and mask together.
- Converts image and mask arrays from `HWC` layout to `CHW` tensor layout.
- Normalizes the image tensor to the `[0, 1]` range.
- Converts the mask tensor into binary foreground-background values.

The resulting tensors match the shapes expected by the model and loss functions. A batch sampled from the training loader has image shape `torch.Size([16, 3, 320, 320])` and mask shape `torch.Size([16, 1, 320, 320])`.

## Model Architecture

The project defines a `SegmentationModel` class that wraps a U-Net architecture from `segmentation_models_pytorch`.

The model configuration is:

- Architecture: `smp.Unet`
- Encoder: `timm-efficientnet-b0`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`
- Output activation inside model: `None`

Using `activation=None` makes the model return raw logits instead of probabilities. This is appropriate because the training objective includes `BCEWithLogitsLoss`, which internally applies the sigmoid operation in a numerically stable way.

The U-Net design combines an encoder-decoder structure with skip connections. The EfficientNet-B0 encoder extracts hierarchical visual features from the input image, while the decoder reconstructs a dense spatial prediction map. The final output is a single-channel logit tensor representing the predicted human foreground score for each pixel.

## Loss Function and Optimization

The model is optimized with a combined segmentation loss:

```python
DiceLoss(mode="binary") + BCEWithLogitsLoss()
```

`DiceLoss` directly targets mask overlap, which is important for segmentation quality because it measures agreement between predicted and ground-truth foreground regions. `BCEWithLogitsLoss` provides pixel-wise binary classification supervision and is well suited for raw logits. Combining both losses gives the model feedback at the region level and the pixel level.

The optimizer is Adam:

- Optimizer: `torch.optim.Adam`
- Learning rate: `0.003`

Adam is used to update all model parameters based on gradients computed during backpropagation.

## Training Strategy

Training is performed for `25` epochs on the CUDA device. The notebook defines separate functions for training and validation:

- `train_fn`: Enables training mode, computes the loss for each batch, performs backpropagation, updates parameters, and returns average training loss.
- `eval_fn`: Enables evaluation mode, disables gradient computation with `torch.no_grad()`, computes validation loss, and returns average validation loss.

During each epoch, the model first iterates through the training loader, then evaluates on the validation loader. The notebook tracks `best_valid_loss`, initialized to infinity, and saves the model state dictionary to `best_mode.pt` whenever validation loss improves.

The best observed validation loss occurs at epoch `15`:

- Epoch `15` training loss: `0.14342068284749984`
- Epoch `15` validation loss: `0.14957228675484657`

The final epoch logs are:

- Epoch `25` training loss: `0.09417678167422612`
- Epoch `25` validation loss: `0.18382345139980316`

The training curve shows that the model learns quickly during the first few epochs, with validation loss dropping from `2.473999261856079` at epoch `1` to `0.19941697269678116` by epoch `5`. Although training loss continues decreasing through epoch `25`, the best validation checkpoint is saved earlier, which helps retain the model state with the strongest validation performance observed in the notebook.

## Inference Pipeline

Inference is performed on selected validation samples after loading the saved checkpoint:

```python
model.load_state_dict(torch.load("/content/best_mode.pt"))
```

For each validation example, the notebook retrieves an image and mask from `validset`, adds a batch dimension with `unsqueeze(0)`, moves the image tensor to the CUDA device, and passes it through the model. The output logits are converted to probabilities with sigmoid:

```python
pred_mask = torch.sigmoid(logits_mask)
```

The probability map is then thresholded at `0.5`:

```python
pred_mask = (pred_mask > 0.5) * 1.0
```

This produces a binary predicted mask where pixels above the threshold are treated as human foreground and pixels below the threshold are treated as background. The notebook visualizes the image, ground-truth mask, and predicted mask using `helper.show_image`.

## Results and Observations

- The notebook demonstrates that a pretrained U-Net can learn the human segmentation task effectively from a relatively small image-mask dataset. The model improves substantially during early training, and the validation loss reaches its best recorded value of `0.14957228675484657` at epoch `15`.

- The inference visualizations compare predicted masks against ground-truth masks for multiple validation indices, including `38`, `3`, `14`, `18`, `35`, and `49`. These visual checks provide qualitative confirmation that the trained model is producing foreground masks from RGB input images.

- The recorded training history and validation visualizations show the complete behavior of the implemented pipeline: loss-based model selection during training and qualitative mask comparison during inference. Together, these outputs demonstrate how the trained U-Net converts RGB validation images into binary foreground masks that can be compared directly with the provided ground-truth annotations.
