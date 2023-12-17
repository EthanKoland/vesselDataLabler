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
* Currently the size buttons do not work

# Notes on the Frangi Filter - FrangiFilter/frangiFilter2DTest.py
* Even with configuring the different parameter there is still poor predictions
* Possibly look into denoising the filter
* Look into other methods to improve the filter