import numpy as np
import matplotlib.pylab as plt

def equal_probability_histogram(x, N_bins=10):
    # compute some quantiles of the data to determine the bin edges
    bin_edges = np.percentile(x, np.linspace(0, 100, N_bins+1))

    n, _ = np.histogram(x, bin_edges)

    f = n / np.diff(bin_edges) / len(x)

    return f, bin_edges

def test_normal():
    # generate some normal data
    N = 1000
    x = np.random.normal(loc=0, scale=1, size=N)

    plt.figure()

    f_orig, bin_edges_orig = np.histogram(x, density=True)

    f, bin_edges = equal_probability_histogram(x, 10)
    plt.step(bin_edges, np.hstack((f, 0)), where='post', label="Equal Count")
    plt.step(bin_edges_orig, np.hstack((f_orig, 0)), where='post', label="Original")
    plt.legend()
    plt.show()

def test_uniform():
    # generate some normal data
    N = 1000
    x = np.random.uniform(low=0, high=5, size=N)

    plt.figure()

    f_orig, bin_edges_orig = np.histogram(x, density=True)

    f, bin_edges = equal_probability_histogram(x, 10)
    plt.step(bin_edges, np.hstack((f, 0)), where='post', label="Equal Count")
    plt.step(bin_edges_orig, np.hstack((f_orig, 0)), where='post', label="Original")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_normal()
    test_uniform()