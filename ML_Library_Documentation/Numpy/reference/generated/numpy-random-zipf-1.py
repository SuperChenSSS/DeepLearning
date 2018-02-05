# Draw samples from the distribution:

a = 2. # parameter
s = np.random.zipf(a, 1000)

# Display the histogram of the samples, along with
# the probability density function:

import matplotlib.pyplot as plt
from scipy import special

# Truncate s values at 50 so plot is interesting:

count, bins, ignored = plt.hist(s[s<50], 50, normed=True)
x = np.arange(1., 50.)
y = x**(-a) / special.zetac(a)
plt.plot(x, y/max(y), linewidth=2, color='r')
plt.show()
