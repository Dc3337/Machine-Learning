import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        
        indices = np.random.randint(X.shape[0], size=self.num_clusters)
        while len(set(indices)) < self.num_clusters:
            indices = np.random.randint(X.shape[0], size=self.num_clusters)
        self.cluster_centers = X[indices, :]
        # print(self.cluster_centers.shape)
        
        # Initialize cluster center and distances
        cluster_index = np.zeros(X.shape[0])
        distance = np.zeros((X.shape[0], self.num_clusters))
        # print(distance.shape)
        
        for _ in range(max_iter):
            # Assign each sample to the closest prototype
            for j in range(self.num_clusters):
                distance[:, j] = np.sqrt(np.sum((X - self.cluster_centers[j])**2, axis=1))
            
            cluster_index= np.argmin(distance, axis=1)
            
            # Update prototypes
            for j in range(self.num_clusters):
                self.cluster_centers[j, :] = np.mean(X[cluster_index == j, :], axis=0)

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        cluster_index = np.zeros(X.shape[0])
        distance = np.zeros((X.shape[0], self.num_clusters))
        
        # Assign each sample to the closest prototype
        for j in range(self.num_clusters):
            distance[:, j] = np.sqrt(np.sum((X - self.cluster_centers[j])**2, axis=1))
            
        cluster_index = np.argmin(distance, axis=1)

        return cluster_index
        # raise NotImplementedError
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        cluster_index = self.predict(X)
        for j in range(self.num_clusters):
            X[cluster_index == j, :] = self.cluster_centers[j,:]
        
        return X 
        # raise NotImplementedError