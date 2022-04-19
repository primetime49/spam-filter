from sklearn.neighbors import KNeighborsClassifier
import extras

class KNN:
    def __init__(self, dataset):
        self.dataset = dataset

    def perform(self, k, n_neighbors):
        acc = 0
        for k in range(0,k):
            target = self.dataset[:, 57]
            features = self.dataset[:, :54]
            X_train,y_train,X_test,y_test = extras.split_data(k,len(self.dataset),features,target)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train,y_train)
            res = clf.score(X_test,y_test)
            acc += res
            print('Fold '+str(k+1)+': ' + str(round(res*100,3)) + '%')
        print('-----')
        print('Accuracy: ' + str(round(acc/10*100,3)) + '%')