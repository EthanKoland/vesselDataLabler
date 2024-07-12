# GOALS
1. frangiFilter2D depend on hessian.py
2. Test frangiFilter2D, please use frangiFilter2DTest. This example also binarizes the result image use adaptive thresholding.
3. FilterVisual.py is for visualizing the Gaussian derivative filters
4. ImageTest.py is for testing Gaussian derivative filters.
5. TestModelFilter.py is for designing pre-defined filter CNN.

# USAGE of canvasClass.py
* first 2 blocks on the upper left corner are for drawing and erasing, white is draw and black is erase.
* the next 3 blocks is for selection the size of the brush. The top most one is the smallest and the bottom most one is the largest.
* to undo small changes, press the q arrow key and the w key for redo changes.
* to undo large changes, press the z arrow key and the x key for redo changes.
* The side buttons currenlty work to save and load the image

# TODO
* Have a shadow of the next and previous images
* Create a version 2 of the segments
* Clean up the repo, add some documentation
* Make the controls more intuitive and easier to understand

# Notes on the Frangi Filter - FrangiFilter/frangiFilter2DTest.py
* Even with configuring the different parameter there is still poor predictions
* Possibly look into denoising the filter
* Look into other methods to improve the filter

## Installation
To run this project, you need to install the required libraries. You can install them using `pip`.

### Required Libraries
- `numpy`
- `opencv-python`
- `tensorflow`
- `pandas`
- `tqdm`
- `matplotlib`
- `Pillow`
- `scikit-image`

Run the following command to install these libraries:
```sh
pip install numpy opencv-python tensorflow pandas tqdm matplotlib Pillow scikit-image
```

## How to Use

1. **Download the folder onto your machine:**

    - Click on the provided link or use a terminal command to clone the repository:
      ```sh
      git clone https://github.com/EthanKoland/vesselDataLabler.git
      ```

2. **Open the folder in Visual Studio Code (or any other IDE of your choice):**

    - Open Visual Studio Code.
    - Select `File > Open Folder` and choose the downloaded folder.



3. **Open the terminal inside the folder:**

    - Navigate to the folder in the terminal if not already there:
      ```sh
      cd path/to/your/folder
      ```

4. **Run the command to install the required libraries:**

    - Run the following command:
      ```sh
      pip install numpy opencv-python tensorflow pandas tqdm matplotlib Pillow scikit-image
      ```

6. **Open the `FrangiFilter` folder:**

    - In the terminal, navigate to the `FrangiFilter` folder:
      ```sh
      cd FrangiFilter
      ```

7. **Run the file `AI_Labeler.py` to open the GUI:**

    - In the terminal, execute the Python script:
      ```sh
      python AI_Labeler.py
      ```

8. **Adjust the parameters to your preference and ensure the masks are being saved in the correct directory:**

    - Use the GUI to set the `FilterLocation` and `OutputLocation`.
    - `FilterLocation` is where the AI filter masks will be saved.
    - `OutputLocation` is where your labels will be saved.

9. **To label the data, load the folder with the raw images:**

    - In the GUI, load the folder containing your raw images.
    - Use the tools provided in the GUI to label the data.

## GUI Guide

- **Pencil Icon:** Select drawing mode.
- **Eraser Icon:** Select eraser mode.
- **Scroll Wheel:** Adjust the size of the brush.
- **Next Image:** Load the next image in the folder (also saves any changes made).
- **Previous Image:** Load the previous image in the folder (also saves any changes made).
- **Filter Image:** Apply the filter to the current image and save it in the directory.
- **Filter Folder:** Apply the filter to every image in the folder.
- **Opening:** Thin out the image to help fill in gaps.
- **Closing:** Widen the image to help fill in gaps.
- **Erode:** Shrink the image in certain places.
- **Dilation:** Widen the image in certain places.
