import numpy as np
import matplotlib.pyplot as plt

def sample_x(x_range, n):
    """sample n x values from a range"""

    x = np.random.uniform(x_range[0], x_range[1], n)

    return x


def sample_y(x, m, c, sigma):
    """make a noisy line around y = mx + c
    standard deviation is sigma
    """

    y = m*x + c + np.random.normal(0.0, scale=sigma, size=x.size)

    return y


def add_outliers(y, m, sigma):
    """Add outliers to y data
    """

    for ii in range(len(y)):

        r = np.random.uniform()

        # 15 percent chance of being an outlier
        if r < 0.15:
            y[ii] = y[ii] + 0.5*y[ii] + 3*m*sigma

    return y


def line_of_best_fit(x, y):
    """get gradient and intercept of line of best fit for x and y data
    """

    z = np.polyfit(x, y, 1)

    return z


def setup(x_range, n, m, c, sigma):
    """create a dataset
    """

    # sample x coordinates
    x = sample_x(x_range,  n)

    # sample noisy y coordinates
    y = sample_y(x, m, c, sigma)

    # add  outliers to dataset
    y = add_outliers(y, m, sigma)

    return np.array(x), np.array(y)


def count_inliers(x, y, sample_model, t):
    """count number of coordinates x, y that lie within a
    distance 't' of a line of best fit defined by sample model
    """

    m, c = sample_model

    test_array = abs(y - m*x - c)

    x_in = x[test_array < t]
    y_in = y[test_array < t]

    inlier_count = len(x_in)

    return inlier_count, x_in, y_in


def plot_inliers(x, y, x_s, y_s,  x_in, y_in, x_not_s, y_not_s, sample_model,
                 t, T, target_m, target_c):
    """Plotting function"""

    m, c = sample_model

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_x = np.array(sorted(x))

    ax.scatter(x, y, marker='.', alpha=0.5, label='Data set')

    ax.fill_between(sorted_x, m*sorted_x + c - t, m *
                    sorted_x + c + t, alpha=0.5, label="Inlier region")

    ax.scatter(x_in, y_in, marker='.', color='g',
                alpha=0.5, label='Sample Model Inliers')
    ax.scatter(x_s, y_s, color='orange', label='Samples', alpha=1.0)

    ax.plot(x, m*x + c, color='r',
            linewidth='2', label='Sample LOBF')

    ax.legend(loc='upper left')

    title_str = "Actual LOBF: y = %.0fx + %.0f\nSample LOBF: y = %.2fx + %.2f\nSample size = %d, Samples inliers = %d (Threshold = %d)" % (
        target_m, target_c, m, c, len(x_s), len(x_in), T)

    ax.set_title(title_str)

    plt.tight_layout()
    plt.show()


def main():

    np.random.seed(1)

    x_range = [0, 10]  # max and minimum range for x values
    n = 1000  # number of samples in dataset
    m, c = 3, 1
    sigma = 1.0

    # set up the data set
    x, y = setup(x_range, n, m, c, sigma)

    # set up RANSAC parameters
    N_trials = 50
    s = 10  # number of subsamples
    t = 2  # threshold of model
    T = 0.75*n  # break if we have reached this threshold

    best_model = []

    best_inlier_count = 0

    for trial in range(N_trials):

        # randomly select s data points
        sample_ids = np.random.choice(range(n), s, replace=False)

        not_sample_ids = sorted(list(set(range(n)) - set(sample_ids)))

        # split data set into sampled and not sampled
        x_s, y_s = x[sample_ids], y[sample_ids]
        x_not_s, y_not_s = x[not_sample_ids], y[not_sample_ids]

        # fit line of best fit to sample
        sample_model = line_of_best_fit(x_s, y_s)

        # count number of inliers among points not sampled
        inlier_count, x_in, y_in = count_inliers(
            x_not_s, y_not_s, sample_model, t)

        # show inliers and threshold decision boundary
        plot_inliers(x, y, x_s, y_s, x_in, y_in, x_not_s, y_not_s,
                     sample_model, t, T, m, c)

        # record if best fit so far
        if inlier_count > best_inlier_count:

            best_model = sample_model
            best_inliers = [x_in, y_in]
            best_inlier_count = inlier_count
            print "New best model found:\nm=%0.3f, c=%0.3f" % (
                best_model[0], best_model[1])

            print "- - - - - - - - - - - - - - - - -"

            # break depending on inlier count
            if inlier_count > T:
                print "Inlier threshold reached!"
                break

    # re-estimate model using all points in inliers
    output_model = line_of_best_fit(best_inliers[0], best_inliers[1])

    print "best model: m = %.2f, c = %.2f" % (output_model[0], output_model[1])

if __name__ == '__main__':
    main()
