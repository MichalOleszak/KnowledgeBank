# networkx v.1.11
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
from itertools import combinations

# Intro ----------------------------------------------------------------------------------------------------------------
# Load Twitter network
T = nx.read_gpickle("data/ego-twitter.p")
# Graph size
len(T.nodes())
# First edge
T.edges(data=True)[0]
# Plot network
nx.draw(T)
plt.show()

# Queries on graphs - extracting nodes and edges of interest
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]

# Types of grahps
undirected = nx.Graph()
directed = nx.DiGraph()
multiedge = nx.MultiGraph()
multiedge_directed = nx.MultiDiGraph()

# Specifying a weight on edges
# Weights can be added to edges in a graph, typically indicating the "strength" of an edge.
# In NetworkX, the weight is indicated by the 'weight' key in the metadata dictionary.
T.edge[1][10]
# Set the 'weight' attribute of the edge between node 1 and 10 of T to be equal to 2
T.edge[1][10]['weight'] = 2
T.edge[1][10]
# Set the weight of every edge involving node 293 to be equal to 1.1
for u, v, d in T.edges(data=True):
    if 293 in [u, v]:
        T.edge[u][v]['weight'] = 1.

# Checking whether there are self-loops in the graph
T.number_of_selfloops()

def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []
    for u, v in G.edges():
        if u == v:
            nodes_in_selfloops.append(u)
    return nodes_in_selfloops

# Check whether number of self loops equals the number of nodes in self loops
# (assert throws an error if the statement evaluates to False)
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))


# Network visualization ------------------------------------------------------------------------------------------------
# Matrix plot
m = nv.MatrixPlot(T)
m.draw()
plt.show()
# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)
# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()

# Circos plot
c = nv.CircosPlot(T)
c.draw()
plt.show()

# Arc plot
a = ArcPlot(T, node_order='category', node_color='category')
a.draw()
plt.show()


# Important nodes ------------------------------------------------------------------------------------------------------
# The number of neighbors that a node has is called its "degree".
# Degree centrality = (no. neighbours) / (no. all possible neighbours) -> N if self-loops allowed, N-1 otherwise

# Get number of neighbours of node 1
T.neighbors(1)

# A function that returns all nodes that have m neighbors
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()
    for n in G.nodes():
        if len(G.neighbors(n)) == m:
            nodes.add(n)
    return nodes

# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)

# Compute degree over the entire network
degrees = [len(T.neighbors(n)) for n in T.nodes()]
# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)
# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()
# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist(list(degrees))
plt.show()
# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.show()

# Graph algorithms for path finding
# Breadth-first search (BFS) algorithm - you start from a particular node and iteratively search through
# its neighbors and neighbors' neighbors until you find the destination node.

def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]
    for node in queue:
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break
        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])
        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))
            return False

# Shortest paths - set of paths where each of them is shortest between some two nodes, for all nodes
# Betweenness centrality = (no. shortest paths that run through the node) /(all possible shortest paths)
# Betweenness centrality captures bottlenecks in the graph
#
# Betweenness centrality is a node importance metric that uses information about the shortest paths in a network.
# It is defined as the fraction of all possible shortest paths between any pair of nodes that pass through the node.
bet_cen = nx.betweenness_centrality(T)
deg_cen = nx.degree_centrality(T)
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))
plt.show()

# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):
    deg_cent = nx.degree_centrality(G)
    max_dc = max(list(deg_cent.values()))
    nodes = set()
    for k, v in deg_cent.items():
        if v == max_dc:
            nodes.add(k)
    return nodes

# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)


# Cliques and communities ----------------------------------------------------------------------------------------------
# Identifying triangle relationships
 def is_in_triangle(G, n):
     """
     Checks whether a node `n` in graph `G` is in a triangle relationship or not.
     Returns a boolean.
     """
     in_triangle = False
     # Iterate over all possible triangle relationship combinations
     for n1, n2 in combinations(G.neighbors(n), 2):
         # Check if an edge exists between n1 and n2
         if G.has_edge(n1, n2):
             in_triangle = True
             break
     return in_triangle


# Finding nodes involved in triangles
# NetworkX provides an API for counting the number of triangles that every node is involved in: nx.triangles(G).
# It returns a dictionary of nodes as the keys and number of triangles as the values.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])
    for n1, n2 in combinations(G.neighbors(n), 2):
        if G.has_edge(n1, n2):
            triangle_nodes.add(n1)
            triangle_nodes.add(n2)
    return triangle_nodes


# Finding open triangles
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
            in_open_triangle = True
            break
    return in_open_triangle


# Compute the number of open triangles in T
num_open_triangles = 0
for n in T.nodes():
    if node_in_open_triangle(T, n):
        num_open_triangles += 1
print(num_open_triangles)


# Finding all maximal cliques of size "n"
# Maximal cliques are cliques that cannot be extended by adding an adjacent edge, and are a useful property of the graph
# when finding communities. NetworkX provides a function that allows you to identify the nodes involved in each maximal
# clique in a graph: nx.find_cliques(G).
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs

# Check that there are 33 maximal cliques of size 3 in the graph T
assert len(maximal_cliques(T, 3)) == 33


# Subgraphs ------------------------------------------------------------------------------------------------------------
# There may be times when you just want to analyze a subset of nodes in a network. To do so, you can copy them out into
# another graph object using G.subgraph(nodes), which returns a new graph object (of the same type as the original
# graph) that is comprised of the iterable of nodes that was passed in.

nodes_of_interest = [29, 38, 42]

def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []
    for n in nodes_of_interest:
        nodes_to_draw.append(n)
        for nbr in G.neighbors(n):
            nodes_to_draw.append(nbr)
    return G.subgraph(nodes_to_draw)


T_draw = get_nodes_and_nbrs(T, nodes_of_interest)
nx.draw(T_draw)
plt.show()