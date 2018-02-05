x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
z = np.polyfit(x, y, 3)
z
# array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])

# It is convenient to use `poly1d` objects for dealing with polynomials:

p = np.poly1d(z)
p(0.5)
# 0.6143849206349179
p(3.5)
# -0.34732142857143039
p(10)
# 22.579365079365115

# High-order polynomials may oscillate wildly:

p30 = np.poly1d(np.polyfit(x, y, 30))
# /... RankWarning: Polyfit may be poorly conditioned...
p30(4)
# -0.80000000000000204
p30(5)
# -0.99999999999999445
p30(4.5)
# -0.10547061179440398

# Illustration:

import matplotlib.pyplot as plt
xp = np.linspace(-2, 6, 100)
_ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2,2)
# (-2, 2)
plt.show()
