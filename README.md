# RANSAC example

Python example of RANSAC algorithm for fitting 2D line of best fit

## Motivation
Using least squares algorithms to fit model parameters can produce poor model fit when there are

A commonly used technique for fitting model parameters is using the least squares algorithm. However, least squares algorithms are very sensitive to outliers, which can lead to poor model fit on such data sets. The RANSAC algorithm overcomes this issue by using sampling to find the model parameters which explain as much of the data as possible. 

## Requirements
- Python
- Numpy
- Matplotlib

## References
- [Wikipedia entry](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [EGGN 512 - Lecture 27-1 RANSAC](https://www.youtube.com/watch?v=NKxXGsZdDp8)
