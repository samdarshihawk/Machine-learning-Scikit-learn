
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)


print(digits.images.shape)
print(digits.data.shape)


plt.imshow(digits.images[1704], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
