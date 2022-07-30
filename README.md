# Gaze Prediction in Natural Videos using End-to-End Deep Learning
Model implementation of Yannic Spreen-Ledebur Master's thesis on dynamic gaze prediction.

This repository contains the model implementation in Pytorch Lightning as well as multiple functionalities for gaze visualization, prediction evaluation and data transformation.

## Expected data structure and format
For model training, the data is expected to be organized as follows
```
data
└───train
│   │
│   └───label_data
│   │   │
│   │   └───video1
│   │   |   |   observer1_video1.txt
│   │   |   |   observer2_video1.txt
│   │   |
│   │   └───video2
│   │       |   observer1_video2.txt
│   │       |   observer2_video2.txt
│   │
│   └───video_data
│       │
│       └───video1
│       |   |   frame_00000.png
│       |   |   frame_00001.png
│       |   |   ...
│       │
│       └───...
│   
└───val
│   │
│   └───label_data
│   │
│   └───video_data
```

`label_data` contains the ground truth labels for each observer and each video. Each row in the label file contains `frame_id`, `em_phase`, `gaze_x` and `gaze_y`, separated by spaces. Eye movement phases can be `0,1,2,3` corresponding to `UNKNOWN/NOISE, FIXATION, SACCADE, SMOOTH PURSUIT`.

`video_data` contains the video data for each video. We extracted frames from the videos and stored them as individual frame images. The data loader theoretically also supports video files, however frame indices are then not read out which we require for further calculations.

<br>
## Usage Examples
#### 1. Training the model
In order to to train the model, the `train_model` function in `model.py` can be used.

Pass a `data_path`, a `clip_duration` in seconds, a `batch_size`, `num_workers` and set the desired model parameters.

```python
# Example training call
train_model(data_path, clip_duration, batch_size, num_workers, out_channels,
                lr=1e-6, backbone_model='mobilenet_v3_large',
                rim_hidden_size=400, rim_num_units=6, rim_k=4)
```

The training checkpoints and Lightning hyperparameters will be saved by Pytorch Lightning in `data/lightning_logs`.

For modifications in the clip sampling, the data loader in `gaze_video_data_module.py` can be modified.

<br>
#### 2. Sampling from a trained model
In order to sample predictions from a trained model and optionally evaluate and visualize these, the function `sample_model` in `sample_model.py` can be utilised.

```python
sample_model(checkpoint_path, output_dir, data_partition, clip_duration, mode='train',
                 calc_metrics=False, show_saliency=True, plot_gaze_change_histograms=True,
                 teacher_forcing_on_inference=False)
```

`sample_model` expects to work with the data partitions described in the thesis and in the function description. It differentiates between sampling from the training set and the validation set based on `mode`.

To use `sample_model` to visualize model predictions over 2 samples, set `calc_metrics=False`. The ground truth labels and model predictions will be overlaid over the sampled clips and saved as a video. NSS scores will also be calculated.

To use `sample_model` to evaluate predictive performance quantitatively over 100 random samples, set `calc_metrics=True`. NSS scores and gaze change distributions are calculated over this set of samples.

<br>
#### 3. Visualizing gaze
In order to visualize gaze ground truth labels and model predictions over the video data, `plot_frames_with_labels` in `utils.py` can be used. It visualizes video frames with bounding boxes for gaze data points.

```python
# Plot a ground truth with a prediction
plot_frames_with_labels(frames, gaze_labels, em_type_labels, [gaze_predictions], [em_type_predictions])
```

To plot ground truth data from the GazeCom dataset, use `plot_gazecom_frames_with_labels` or `plot_gazecom_frames_with_all_observers` from `utils.py`.
```python
# Plot ground truth with one observer
plot_gazecom_frames_with_labels(video_path: str, label_path: str, raw_label_path: str, save_to_directory=None)

# Plot ground truth with many observers
plot_gazecom_frames_with_all_observers(video_path: str, label_dir: str, plot_em_data=False, save_to_directory=None, n_observers=None)
```


<br>
## Feature branches
- `change_prediction`: Instead of absolute gaze, train the model to predict frame-wise gaze changes instead. This is hoped to counter-act center bias in predictions, but training results were very poor in our case.
- `positional_encoding`: To enhance the spatial information given to the RIM units, positional encoding can be applied to the features extracted in the Feature Pyramid Network. We do this through a learned additive encoding. In tests, we could not see a substantial improvement in model performance however.
- `seeding_hidden_state`: Pass observer to RIMs to generate observer-specific RIM initial states. This allows the model to differentiate between different observers. 
