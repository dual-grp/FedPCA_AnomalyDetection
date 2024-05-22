# PCA transform with zero mean
def self_pca_transform_with_zero_mean(X_train, V_k):
  return (X_train).dot(V_k)
  
# PCA inverse transform with zero mean
def self_inverse_transform_with_zero_mean(X_pca, V_k):
  return (X_pca.dot(V_k.T))