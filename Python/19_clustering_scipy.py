from scipy.cluster.vq import kmeans, vq
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten


# Basics ---------------------------------------------------------------------

# k-means in ScyPy
centroids,_ = kmeans(df, 2)
df['cluster_labels'], _ = vq(df, centroids)
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()

# hclust in ScyPy
Z = linkage(df, 'ward')
df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()

# Data has to be normalized before clustering!
# If units of features are not comparable, clustering results are biased!
scaled_data = whiten(data)


# Hierarchical clustering ----------------------------------------------------

# method: how to calculate the proximity of clusters
#   - single: based on two closest objects
#   - complete: based on the two furthest objects
#   - average: based on the arithmetic mean of all objects
#   - centroid: based on the geometric mean of all objects
#   - median: based on the median of all objects
#   - ward: based on the sum of squares (sos of joint clusters - individual sos)

# metric: distance metric
