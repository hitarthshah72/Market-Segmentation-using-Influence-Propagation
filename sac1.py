# Name: Hitarth Himanshu Shah
# Unity: hshah4

import sys
import csv
import random
import math
import pandas as pd
import numpy
from igraph import *
from scipy import spatial


def cosine_similarity() :
    cossim = [[0 for x in range(vertices)] for x in range(vertices)]
    for i in range(0, vertices) :
        vertex_i = graph.vs.select(i)[0].attributes().values()
        for j in range(i, vertices) :
            vertex_j = graph.vs.select(j)[0].attributes().values()
            distance = spatial.distance.cosine(list(vertex_i), list(vertex_j)) + 1.0
            cossim[i][j] = 1.0 / (distance)
            cossim[j][i] = cossim[i][j]


def qneuman(x, comm, nod, num_edges):
    return x - sum(graph.degree(comm)) * graph.degree(nod) / (2 * num_edges)

def qattr(g_attr, cossim, comm, nod):
    for item in comm:
        g_attr = g_attr + cossim[item][nod]
    return g_attr / len(comm) / len(comm)

def compute_modularity_gain(nod, comm):
    x = 0
    deg = 0
    num_edges = len(graph.es)
    comm = list(set(comm))
    for item in comm:
    	if graph.are_connected(nod, item):
    		ind = graph.get_eid(nod, item)
    		x += graph.es["weight"][ind]
    dq_neuman = qneuman(x, comm, nod, num_edges)
    q_attr = 0
    dq_attr = qattr(q_attr, cossim, comm, nod)
    return alpha * dq_neuman + (1 - alpha) * dq_attr



def make_community(graph, community):
    count = 0
    for item in range(vertices):
        gain = []
        curr_community = []
        for v in community:
            if item in v:
                curr_community = v

        max_gain = -1
        max_community = []

        for com in community:
            gain = compute_modularity_gain(item, com)
            if gain > 0:
                if gain > max_gain:
                    max_gain = gain
                    max_community = com

        if set(curr_community) != set(max_community):
            if max_gain > 0:
                curr_community.remove(item)
                max_community.append(item)
                count += 1
                if len(curr_community) == 0:
                    community.remove([])
    return count


def phaseone(graph, cossim, community):
	cossim = cosine_similarity()
	convergence = make_community(graph, community)
	i = 0
	while convergence and i < 15:
		i+=1
		cnt = make_community(graph, community)


def phasetwo(graph, cossim, mapped_communities, mapped_vertices):
	global vertices
	nv = 0
	for community in mapped_communities :
		for v in community :
			mapped_vertices[v] = nv
		nv += 1

	graph.contract_vertices(mapped_vertices, combine_attrs = "mean")
	graph.simplify(multiple = True, loops = True)

	vertices = nv
	mapped_communities = [[x] for x in range(vertices)]
	graph.es["weight"] = [0 for x in range(len(graph.es))]

	for edge in edges :
		left_comm = mapped_vertices[edge[0]]
		right_comm = mapped_vertices[edge[1]]

		if left_comm != right_comm:
			id = graph.get_eid(left_comm, right_comm)
			graph.es["weight"][id] += 1

	cossim = cosine_similarity()
	phaseone(graph, cossim, mapped_communities)

def create_file(communities):
    op_file = open("./communities.txt", "w")
    for item in communities:
        for i in range(len(item)):
            if i != 0:
                op_file.write(",")
            op_file.write(str(item[i]))
        op_file.write("\n")
    op_file.close()


alpha = float(sys.argv[1])

# Extracting vertices
attributes = []
attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')
vertices = len(attributes)

# Extracting edges
edges = []
edgelist_file = open('./data/fb_caltech_small_edgelist.txt')
edge_list = edgelist_file.read().split("\n")
for edge in edge_list:
    v = edge.split(' ')
    if v[0] != '' and v[1] != '':
        edges.append((int(v[0]),int(v[1])))

# Creating graph
graph = Graph()
attributes_names = list(attributes.columns.values)
graph.add_vertices(vertices)
graph.add_edges(edges)
graph.es["weight"] = [1 for x in range(len(edges))]
for attribute in attributes_names :
    graph.vs[attribute] = list(attributes[attribute])

cossim = [[0 for x in range(vertices)] for x in range(vertices)]

communities = [[x] for x in range(vertices)]
mapped_vertices = [0 for x in range(vertices)]

phaseone(graph, cossim, communities)
phasetwo(graph, cossim, communities, mapped_vertices)



create_file(communities)
