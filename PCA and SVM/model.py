import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model

        coVar_mat = np.cov(X.T)
        EigenValues, EigenVectors = np.linalg.eig(coVar_mat)

        top_k_eigenvectors = EigenVectors[:,:self.n_components]
        self.components = top_k_eigenvectors

        # raise NotImplementedError
    
    def transform(self, X) -> np.ndarray:
        # transform the data

        Transformed_Data = np.matmul(self.components.T,X.T).T
        return Transformed_Data
        # raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters

        dim = X.shape[1]
        self.w, self.b = np.zeros(dim), 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            tot_data = X.shape[0]
            index = np.random.choice(tot_data)

            data_X, label_Y = X[index], y[index]

            if(label_Y*(np.dot(self.w,data_X.T) + self.b)<1):
                # self.w = self.w + learning_rate*(C*data_X*label_Y - )
                gradiant_w = self.w - (C*label_Y*data_X)
                gradiant_b = -(C*label_Y)
            else:
                gradiant_w = 0
                gradiant_b = 0
            
            self.w = self.w - learning_rate*(gradiant_w)
            self.b = self.b - learning_rate*(gradiant_b)
            
            # raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return (np.dot(X,self.w) + self.b)
        # raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
        C, learning_rate, num_iters = kwargs.values()

        for i in range(len(self.models)):
            new_label = np.where(y==i,1,-1)
            self.models[i].fit(X,new_label,learning_rate,num_iters,C)

        # raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        tot_test_data = X.shape[0]
        tot_label = self.num_classes

        predicted_labels = np.full((tot_test_data,tot_label), float('-inf'))

        for i in range(len(self.models)):
            predicted_labels[:,i] = self.models[i].predict(X)
        
        Result = np.argmax(predicted_labels,axis=1)
        return Result

        # raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        predicted_label = self.predict(X)
        unique_label_list = np.unique(y)
        precision_list = np.zeros(len(unique_label_list))

        # index = 0
        for i in unique_label_list:
            true_positive = np.sum((y == i) & (predicted_label == i))
            false_positive = np.sum((y != i) & (predicted_label == i))
            tot_denominator = true_positive + false_positive
            if(tot_denominator == 0):
                precision_list[i] = 0
            else:
                precision_list[i] = (true_positive)/(tot_denominator)
            # index += 1

        return (sum(precision_list)/len(unique_label_list))
        # raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        predicted_label = self.predict(X)
        unique_label_list = np.unique(y)
        recall_list = np.zeros(len(unique_label_list))

        index = 0
        for i in unique_label_list:
            true_positive = np.sum((y == i) & (predicted_label == i))
            false_negative = np.sum((y == i) & (predicted_label != i))
            tot_denominator = true_positive + false_negative
            if(tot_denominator == 0):
                recall_list[index] = 0
            else:
                recall_list[index] = (true_positive)/(tot_denominator)
            index += 1

        return (sum(recall_list)/len(unique_label_list))
        # raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        precisionScore = self.precision_score(X,y)
        recallScore = self.recall_score(X,y)

        f1Score = (2*precisionScore*recallScore)/(precisionScore+recallScore)
        return f1Score
        # raise NotImplementedError
