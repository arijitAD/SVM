from sklearn import svm
import csv
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# global variables
class1 = []
class2 = []
labels = []

def buildVectors():
    st = np.transpose([class1, class2, labels])
    return st

def load_file(filename):
    print "\nloading raw data from {filename}".format(filename=filename)
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', )
        reader.next()  # skip the header line
        for row in reader:
            class1.append(float(row[0]))
            class2.append(float(row[1]))
            labels.append(float(row[2]))
    return class1, class2, labels

if __name__ == '__main__':
    # load the raw data
    tup = load_file('DATA_SVM.csv')
    data = buildVectors()
    c = [1, 10, 100, 1000, 10000]
    d = [2,3,4]

    poly_max_accuracy = 0
    poly_c = 0
    poly_d = 0

    for i in c:
        for j in d:
            print "\nBuilding Polynomial SVM with C = {c} and Degrees = {d}".format(c=i, d=j)
            accuracy = []
            for val in range(0, 30):
                kf = KFold(n_splits=10, shuffle=True)
                for test, train in kf.split(data):
                    Xtrain, Xtest, Ytrain, Ytest = data[train][:, :2], data[test][:, :2], data[train][:, 2], data[test][                                                                                    :, 2]
                    poly_svc = svm.SVC(C=i, degree=j, kernel='poly')
                    poly_svc.fit(Xtrain, Ytrain)
                    acc = poly_svc.score(Xtest, Ytest)
                    accuracy.append(acc * 100)

            accuracy = np.array(accuracy)
            if np.mean(accuracy) > poly_max_accuracy:
                poly_max_accuracy = np.mean(accuracy)
                poly_svc_final = poly_svc
                poly_c = i
                poly_d = j
            print "Mean = ", np.mean(accuracy)
            print "Standard Deviation = ", np.std(accuracy, dtype=np.float64)

    rbf_max_accuracy = 0
    rbf_c = 0
    rbf_d = 0
    d = [1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 5]

    for i in c:
        for j in d:
            print "\nBuilding RBF SVM with C = {c} and Sigma = {d}".format(c=i, d=j)
            accuracy = []
            for val in range(0, 30):
                kf = KFold(n_splits=10, shuffle=True)
                for test, train in kf.split(data):
                    Xtrain, Xtest, Ytrain, Ytest = data[train][:, :2], data[test][:, :2], data[train][:, 2], data[test][                                                                                         :, 2]
                    rbf_svc = svm.SVC(C=i, kernel='rbf', gamma=j)
                    rbf_svc.fit(Xtrain, Ytrain)
                    acc = rbf_svc.score(Xtest, Ytest)
                    accuracy.append(acc * 100)
            accuracy = np.array(accuracy)
            if np.mean(accuracy) > rbf_max_accuracy:
                rbf_max_accuracy = np.mean(accuracy)
                rbf_svc_final = poly_svc
                rbf_c = i
                rbf_d = j
            print "Mean = ", np.mean(accuracy)
            print "Standard Deviation = ", np.std(accuracy, dtype=np.float64)

h = .02  # step size in the mesh
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with RBF kernel', 'SVC with polynomial kernel']

for i, clf in enumerate(( rbf_svc_final,poly_svc_final )):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.coolwarm)
    plt.xlabel('X - Axis')
    plt.ylabel('Y - Axis')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

print "Max Accuracy (Polynomial) = ", poly_max_accuracy, " C = ",poly_c," D = ",poly_d
print "Max Accuracy (RBF) = ", rbf_max_accuracy, " C = ",rbf_c," D = ",rbf_d

