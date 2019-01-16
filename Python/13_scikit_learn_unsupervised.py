from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, normalize, MaxAbsScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# k-means --------------------------------------------------------------------------------------------------------------
# Fit model to the data
iris = datasets.load_iris().data
model = KMeans(n_clusters=3)
model.fit(iris)
labels = model.predict(iris)

# Plot results with cluster centroids in 2D
xs = iris[:,0]
ys = iris[:,2]
plt.scatter(xs, ys, c=labels, alpha=0.5)
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,2]
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

# Select number of clusters: inertia low, but not too many clusters: look for an elbow in the plot
ks = range(1, 6)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(iris)
    inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Assess quality by looking at the cross-table (only possible if some true labels are available)
species = sorted(datasets.load_iris().target_names.tolist() * 50)
model = KMeans(n_clusters=3)
labels = model.fit_predict(iris)
df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)


# Scaling data before clustering
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, kmeans)
labels = pipeline.fit_predict(iris)
df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

# Normalizing data
# While StandardScaler() standardizes features by removing the mean and scaling to unit variance,
# Normalizer() rescales each sample independently of the other.
normalizer = Normalizer()


# Agglomerative hierarchical clustering --------------------------------------------------------------------------------

# Linkage methods:
#  - complete: the distance between clusters is the distance between the furthest points of the clusters
#  - single: distance between clusters is the distance between the closest points of the clusters

iris = datasets.load_iris().data
hclust = linkage(iris, method='complete')
species = sorted(datasets.load_iris().target_names.tolist() * 50)

# Plot the dendrogram
dendrogram(hclust, labels=species, leaf_rotation=90, leaf_font_size=6)
plt.show()

# Extracting the cluster labels at a specific height of dendrogram (e.g. 3)
labels = fcluster(hclust, 3, criterion='distance')

# t-SNE for 2-dimensional maps (t-distributed stochastic neighbour embedding)
#  - maps samples from high-dimensional space to 2D or 3D to enable visualization
#  - approximately preserves distances between the samples
#  - axis values not interpretable and different in each run!
model = TSNE(learning_rate=200)   # try different values between 50 and 200
tsne_features = model.fit_transform(iris)
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]
species_numbers = datasets.load_iris().target
plt.scatter(xs, ys, c=species_numbers)
plt.show()


# Decorrelating data and dimension reduction ---------------------------------------------------------------------------
iris_petals = datasets.load_iris().data[:, 2:]
length = iris_petals[:, 0]
width = iris_petals[:, 1]
# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()
# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)
print(correlation)

# Decorrelating the petal measurements with PCA
model = PCA()
pca_features = model.fit_transform(iris_petals)
# Scatter plot PCA features
xs = pca_features[:,0]
ys = pca_features[:,1]
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()
# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)
print(correlation)

# The first principal component of the data is the direction in which the data varies the most. Use PCA to find
# the first principal component of the length and width measurements of the iris peta. samples, and represent it as
# an arrow on the scatter plot.
plt.scatter(iris_petals[:,0], iris_petals[:,1])
model = PCA()
model.fit(iris_petals)
mean = model.mean_
first_pc = model.components_[0,:]
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
plt.axis('equal')
plt.show()

# Intrinsic dimension of a dataset = number of features required to approximate it;
# can be identified by looking at PCA features' variance
wine = datasets.load_wine().data
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(wine)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Keeping only important dimensions
pca = PCA(n_components=2)

# TfidfVectorizer transforms a list of documents into a word frequency array, output as a csr_matrix (sparse matrix)
documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
tfidf = TfidfVectorizer()
csr_mat = tfidf.fit_transform(documents)
print(csr_mat.toarray())
words = tfidf.get_feature_names()
print(words)

# TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format (unlike PCA)
# PCA is (truncated) SVD on centered data (by per-feature mean subtraction). If the data is already centered, those
# two classes will do the same. In practice TruncatedSVD is useful on large sparse data sets which cannot be centered
# without making the memory usage explode.

# Cluster some popular pages from Wikipedia
df = pd.read_csv('Data/Wikipedia articles/wikipedia-vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)

svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)
pipeline.fit(articles)
labels = pipeline.predict(articles)
df = pd.DataFrame({'label': labels, 'article': titles})
print(df.sort_values('label'))


# Non-negative matrix factorization (NMF) ------------------------------------------------------------------------------
#  - an interpretable dimensionality reduction technique
#  - only works with non-negative features
#  - number of components has to be chosen beforehand
#  - for documents: NMF components represent topics, features combine topics into documents
#  - for images: NMF components are parts of images
df = pd.read_csv('Data/Wikipedia articles/wikipedia-vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)
file = open("data/Wikipedia articles/wikipedia-vocabulary-utf8.txt", "r")
words = file.read().split("\n")
# Run NMF
model = NMF(n_components=6)
model.fit(articles)
nmf_features = model.transform(articles)
print(nmf_features)
# Check the components
df = pd.DataFrame(nmf_features, index=titles)
print(df.loc['Anne Hathaway'])
print(df.loc['Denzel Washington'])
# For both actors, the NMF feature 3 has by far the highest value. This means that both articles are reconstructed
# using mainly the 3rd NMF component.
# Identify the topics that the articles about Anne Hathaway and Denzel Washington have in common:
components_df = pd.DataFrame(model.components_, columns=words)
component = components_df.iloc[3, :]
print(component.nlargest())

# Use NMF to decompose grayscale images into their commonly occurring patterns
# (100 images, each 13x8, showing LED digital displays)
digits = np.genfromtxt('data/lcd-digits.csv', delimiter=',')
# Show one image
bitmap = digits[0, :].reshape(13, 8)
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
# Show how NMF learns parts of images
def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
model = NMF(n_components=7)
features = model.fit_transform(digits)
for component in model.components_:
    show_as_image(component)
digit_features = features[0, :]

# Which articles are similar to 'Cristiano Ronaldo'?
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
# Select the row corresponding to 'Cristiano Ronaldo'
article = df.loc['Cristiano Ronaldo']
# Calculate the cosine similarity of every row with article
similarities = df.dot(article)
# Display those with the largest cosine similarity
print(similarities.nlargest())

# Building recommender systems using NMF.
# Goal: to recommend popular music artists! Rows of the array artists correspond to artists and columns
# correspond to users. The entries give the number of times each artist was listened to by each user.
artist_names = pd.read_csv('data/Musical artists/artists.csv', header=None).loc[:, 0].tolist()
artists = pd.read_csv('data/Musical artists/scrobbler-small-sample.csv')
artists = artists.pivot_table(index='artist_offset', columns='user_offset', values='playcount', fill_value=0)
# MaxAbsScaler transforms the data so that all users have the same influence on the model,
# regardless of how many different artists they've listened to
scaler = MaxAbsScaler()
nmf = NMF(n_components=20)
normalizer = Normalizer()
pipeline = make_pipeline(scaler, nmf, normalizer)
norm_features = pipeline.fit_transform(artists)
# Suppose you were a big fan of Bruce Springsteen - which other musicial artists might you like?
df = pd.DataFrame(norm_features, index=artist_names)
artist = df.loc['Bruce Springsteen']
similarities = df.dot(artist)
print(similarities.nlargest())

