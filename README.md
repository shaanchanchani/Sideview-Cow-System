# Sideview-Cow-System

This repository hosts an adaptable, library-based implementation of a video analytics system for Cows. It implements He Liu's
post-processing procedure for side-view cow feature extraction. A detailed discussion of this procedure can be found linked here:

[A cow structural model for video analytics of cow health](https://arxiv.org/abs/2003.05903)

The motivation for this overhaul stems from these features having demonstrated success in body weight estimation.

## Installation
Clone this repo, and run
```
conda create -n moo python=3.8.18 
conda activate moo
pip install "deeplabcut[tf]"==2.3.8
```

## Usage

The only function necessary to run the system end-to-end is `run_sideview_system(...)`.

This function requires the following parameters to be specfied:

1) `input_video_folders`: A list of folder names containing your test videos. These folders should be at the same directory level as the library.

2) `output_folder` : Directory to store analysis results. Output video can be found in the `merge_res` sub-folder of this directory.

3) `cow_model_key`: String specifyig which CNN set and cow model to use.


The following keys are supported for `cow_model_key`:

- `channel_3` : the original structural model with DLC CNNs Liu trained using data from just the IP camera

- `merge` : the original structural model with DLC CNNs Liu trained using data from all three cameras (IP, DVR, GoPro) 

- `merge_underbelly` : structural model with an underbelly point with DLC CNNs I trained using data from all three cameras (IP, DVR, GoPro)

## Example
```python 
import sideview_system as ss

ss.run_sideview_system(['test_video_folder'],'./sideview_outputs/','merge_underbelly')
```

