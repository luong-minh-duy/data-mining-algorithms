import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import norm

# Load dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

# Loop through each feature
for feature_idx in range(X.shape[1]):
    
    plt.figure()
    
    # For each class
    for class_idx in np.unique(y):
        feature_values = X[y == class_idx, feature_idx]
        
        mean = np.mean(feature_values)
        std = np.std(feature_values)
        
        x_range = np.linspace(feature_values.min(), 
                              feature_values.max(), 
                              200)
        
        pdf = norm.pdf(x_range, mean, std)
        
        # Histogram
        plt.hist(feature_values, bins=10, density=True, alpha=0.4)
        
        # Gaussian curve
        plt.plot(x_range, pdf)
    
    plt.title(f"Feature Distribution by Class: {feature_names[feature_idx]}")
    plt.xlabel(feature_names[feature_idx])
    plt.ylabel("Density")
    plt.legend(class_names)
    plt.show()