# Draw samples from the distribution:

shape, scale = 2., 1. # mean and width
s = np.random.standard_gamma(shape, 1000000)

# Display the histogram of the samples, along with
# the probability density function:

import matplotlib.pyplot as plt
import scipy.special as sps
count, bins, ignored = plt.hist(s, 50, normed=True)
y = bins**(shape-1) * ((np.exp(-bins/scale))/ \
                      (sps.gamma(shape) * scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()
