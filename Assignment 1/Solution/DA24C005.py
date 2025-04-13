import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# data loading

data=pd.read_csv("Image Data Points.csv", header=None)
data.columns=pd.Index(['x', 'y'])

# cleansing the data

data_trans=data.round(0).drop_duplicates().astype(dtype=int)
sparse_matrix=np.zeros(shape=(1000, 1000))
for idx, coordinate in data_trans.iterrows():
    sparse_matrix[coordinate['x'], coordinate['y']]=1


# Function that converts sparse matrix into dense representation [(x,y) coordinates data frame]

def sparse_to_dense(sparse_matrix):
    row,col = np.nonzero(sparse_matrix)
    return pd.DataFrame({'x': row, 'y': col})

# flip_matrix is the matrix used to flip a matrix horizontally

flip_matrix=np.zeros(shape=(1000, 1000), dtype=int)
for i in range(1000):
    flip_matrix[999-i, i]=1

""" DATA VISUALISATION """

fig, axes=plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(w=10, h=10)
axes[0, 0].scatter(data_trans['x'], data_trans['y'],s=1)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('Original Image')


# Flipping the matrix horizontally

flipped_matrix = np.matmul(sparse_matrix, flip_matrix)
flipped_data = sparse_to_dense(flipped_matrix)

axes[0, 1].scatter(flipped_data['x'], flipped_data['y'],s=1)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title('Flipped Horizontally')

# rotating the matrix by 90 degrees

rot_matrix=np.matmul(sparse_matrix.T, flip_matrix)
rot_data= sparse_to_dense(rot_matrix)

axes[1, 0].scatter(rot_data['x'], rot_data['y'],s=1)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('90 degree rotation')

#Image with Points

plt.subplot(2,2,4)
ig=img.imread("Image.png")
plt.imshow(ig)
axes[1, 1].set_title('Image')

fig.tight_layout()
plt.show()




