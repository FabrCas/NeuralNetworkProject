# import section
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os
import time

DATASET = "MNIST"

#check if cuda is available.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MNIST_SIZE = (28, 28)

# TESTING on SHVN dataset
def get_SVHN(batchSize, data_dir="./"):
    """Returning cifar dataloder."""
    transform = transforms.Compose([transforms.Resize(32), #3x32x32 images.
                                    transforms.ToTensor()])
    train = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    test = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    train_dataloader = DataLoader(train, batch_size= batchSize, shuffle=False)
    test_dataloader = DataLoader(test, batch_size= batchSize, shuffle=False)
    DATASET = "SVHN"
    return train_dataloader,test_dataloader

def get_MNIST(batchSize, data_dir="./"):
    """Returning cifar dataloder."""
    transform = transforms.Compose([transforms.Resize(32), #3x32x32 images.
                                    transforms.ToTensor()])
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train, batch_size=batchSize, shuffle= False)       # shuffle rules the randomness of samples
    test_dataloader = DataLoader(test, batch_size=batchSize, shuffle= False)
    DATASET = "MNIST"
    return train_dataloader,test_dataloader,


def saveModel(KNN):
    print("\nSaving the model....")
    KNN_model = '1NN.pt'
    if not os.path.exists("./prj_1NN_model"):
        os.makedirs("./prj_1NN_model")
    path = F"./prj_1NN_model/{KNN_model}"
    pickle.dump(KNN, open(path, 'wb'))
    # shutil.make_archive('/prj_1NN_model', 'zip', '/prj_1NN_model')
    # files.download('/prj_1NN_model.zip')

def test_image(X_train):
    img = X_train[0].squeeze()
    plt.imshow(img)
    plt.show()

def extract_data(data, size):
  X = []; Y = []
  for i,(xs,labels)  in enumerate(data):  # take batches
    complete_tag = False
    for x in xs:                # take single element
      img =  x.numpy()
      X.append(img)
      if len(X) >= size:
        break
    for label in labels:        # take single label
      Y.append(label.numpy())
      if len(Y) >= size:
        complete_tag = True
        break
    if complete_tag: # we got all the elements, exit
      break
  return  X, Y

def create_learn_1NN():
    KNN = KNeighborsClassifier(n_neighbors=1)

    train_KNN, test_KNN = get_MNIST(batchSize=128)
    # get the size of the dataset
    train_size = train_KNN.dataset.data.shape
    test_size = test_KNN.dataset.data.shape
    test = False
    # print("train size: {}, test size {}".format(train_size, test_size))

    if not(test): # maximum values for the complete evaluation
        train_size = train_size[0]
        test_size = test_size[0]
    else:  # size just the test the classifier
        train_size = 2000
        test_size = 500

    # *******************************************************************************************************
    startTime = time.time()
    print("Extracting data from MNIST....")
    X_train, y_train = extract_data(train_KNN, train_size)
    X_test, y_test = extract_data(test_KNN, test_size)
    print("End extraction phase, time: {} [s]".format(time.time() -startTime))

    test_image(X_test)
    X_train = np.asarray(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    # *******************************************************************************************************
    startTime = time.time()
    print("learning the model....")
    KNN.fit(X_train, y_train)
    print("End learning phase, time: {} [s]".format(time.time() - startTime))
    X_test = np.asarray(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # *******************************************************************************************************
    startTime = time.time()
    print("Making predictions....")
    y_test_pred = KNN.predict(X_test)
    print("End prediction phase, time: {} [s]".format(time.time() - startTime))
    # *******************************************************************************************************
    # print result
    acc = accuracy_score(y_test_pred,y_test)
    print('\n---------------------->Accuracy: %f' % (acc))
    return KNN

if __name__ == "__main__":
    model = create_learn_1NN()
    saveModel(model)