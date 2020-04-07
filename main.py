#!/usr/bin/env python

import sys
import os.path
import re
import json
import math
import numpy as np
from tqdm import tqdm
import igraph as ig
# import infomap

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


_styleColors = [
	"#1f77b4",
	"#ff7f0e",
	"#2ca02c",
	"#d62728",
	"#9467bd",
	"#8c564b",
	"#e377c2",
	"#7f7f7f",
	"#bcbd22",
	"#17becf",
	"#aec7e8",
	"#ffbb78",
	"#98df8a",
	"#ff9896",
	"#c5b0d5",
	"#c49c94",
	"#f7b6d2",
	"#c7c7c7",
	"#dbdb8d",
	"#9edae5",
];

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def check_symmetric(a, rtol=1e-05, atol=1e-08):
	return np.allclose(a, a.T, rtol=rtol, atol=atol)

def isFloat(value):
	if(value is None):
		return False
	try:
		numericValue = float(value)
		return np.isfinite(numericValue)
	except ValueError:
		return False

def loadCSVMatrix(filename):
	return np.loadtxt(filename,delimiter=",")



configFilename = "config.json"
argCount = len(sys.argv)
if(argCount > 1):
		configFilename = sys.argv[1]

outputDirectory = "output"
figuresOutputDirectory = os.path.join(outputDirectory, "figures")

if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

if(not os.path.exists(figuresOutputDirectory)):
		os.makedirs(figuresOutputDirectory)

with open(configFilename, "r") as fd:
		config = json.load(fd)

# "index": "data/index.json",
# "label": "data/label.json",
# "csv": "data/csv",
# "color-property": "degree",
# "size-property":"degree"

indexFilename = config["index"]
labelFilename = config["label"]
CSVDirectory = config["csv"]

colorProperty = "degree"
sizeProperty = "degree"
plotLabels = True
plotNullModels = True

if("color-property" in config):
	colorProperty = config["color-property"].lower()

if("size-property" in config):
	sizeProperty = config["size-property"].lower()

if("plot-labels" in config):
	plotLabels = config["plot-labels"]

if("plot-null-models" in config):
	plotNullModels = config["plot-null-models"]

with open(indexFilename, "r") as fd:
	indexData = json.load(fd)

with open(labelFilename, "r") as fd:
	labelData = json.load(fd)

labels = [entry["name"].replace(".label","") for entry in labelData]


def sortByFrequency(arr):
	s = set(arr)
	keys = {n: (-arr.count(n), arr.index(n)) for n in s}
	return sorted(list(s), key=lambda n: keys[n])

def convertColorToHex(r,g,b,a):
	def roundclamp(x): 
		return int(round(max(0, min(x, 255))))
	return '#%02x%02x%02x'%(roundclamp(r*255),roundclamp(g*255),roundclamp(b*255))

def convertColorToRGBAString(r,g,b,a):
	return "rgba(%d,%d,%d,%f)"%(round(r*255),round(g*255),round(b*255),a)

from matplotlib import collections  as mc
def drawGraph(graph,ax):
	# print(graph.vcount())
	# print("Layouting...");
	#positions = np.array(graph.layout_drl());
	positions = np.array(graph.layout_lgl(maxiter=1000,coolexp = 2.0));
	# print("Plotting...");
	linesX = []
	linesY = []
	segments = []
	positionsX = positions[:,0]
	positionsY = positions[:,1]
	for edge in graph.es:
		source = edge.source
		target = edge.target
		fx = positionsX[source]
		fy = positionsY[source]
		tx = positionsX[target]
		ty = positionsY[target]
		linesX.append(fx)
		linesX.append(tx)
		linesX.append(None)
		linesY.append(fy)
		linesY.append(ty)
		linesY.append(None)
		segments.append([(fx, fy), (tx, ty)])
	# plt.plot(linesX,linesY,alpha=0.1);
	lc = mc.LineCollection(segments, colors=graph.es["color"], linewidths=1.5)
	ax.add_collection(lc)
	# print("Finished Plotting...");
		
	ax.scatter(positionsX,positionsY,marker="o",c=graph.vs["color"],s=graph.vs["vertex_size"],zorder=10);
	vertexColors = graph.vs["color"];
	if("name" in graph.vertex_attributes()):
		for i,label in enumerate(graph.vs["name"]):
			textColor = adjust_lightness(vertexColors[i],0.5)
			text_object = ax.annotate(label, (positionsX[i], positionsY[i]),
				color=textColor,fontsize=4, ha='center',zorder=20)

	ax.autoscale()
	ax.margins(0.01)
	

for entry in indexData:
	entryFilename = entry["filename"]

	alreadySigned = ("separated-sign" in entry) and entry["separated-sign"]

	#inputfile,outputfile,signedOrNot
	filenames = [entryFilename]
	baseName,extension = os.path.splitext(entryFilename)

	if(alreadySigned):
		filenames += [baseName+"_negative%s"%(extension)]

	if("null-models" in entry and plotNullModels):
		nullCount = int(entry["null-models"])
		filenames += [baseName+"-null_%d%s"%(i,extension) for i in range(nullCount)]
		if(alreadySigned):
			filenames += [baseName+"_negative-null_%d%s"%(i,extension) for i in range(nullCount)]

	hasCommunities = False
	if("community" in entry):
		hasCommunities = (entry["community"]==True)
	
	for filename in tqdm(filenames):
		adjacencyMatrix = loadCSVMatrix(os.path.join(CSVDirectory, filename))
		directionMode=ig.ADJ_DIRECTED
		weights = adjacencyMatrix
		if(check_symmetric(adjacencyMatrix)):
			directionMode=ig.ADJ_UPPER
			weights = weights[np.triu_indices(weights.shape[0], k = 0)]
		g = ig.Graph.Adjacency((adjacencyMatrix != 0).tolist(), directionMode)
		weighted = False
		if(not ((weights==0) | (weights==1)).all()):
			g.es['weight'] = weights[weights != 0]
			weighted = True
		graph = g
		graph.vs["degree"] = g.degree()
		graph.vs["indegree"] = g.indegree()
		graph.vs["outdegree"] = g.outdegree()

		sizeArray = graph.vs[sizeProperty]
		maxProperty = max(sizeArray)

		graph.vs["vertex_size"] = [x/maxProperty*200+4 for x in sizeArray]
		
		if(hasCommunities):
			inputBaseName,_ = os.path.splitext(filename)
			communities = [];
			with open(os.path.join(CSVDirectory,"%s_community.txt"%os.path.basename(inputBaseName)), "r") as fd:
				for line in fd:
					communities.append(line.strip());
			graph.vs["Community"] = communities;
						



		if("Community" not in graph.vertex_attributes() or colorProperty!="community"):
			colormap = plt.get_cmap("winter");
			colorPropertyArray = np.array(graph.vs[colorProperty]);
			colorPropertyTransformed = np.log(colorPropertyArray+1.0)
			colorPropertyTransformed -= np.min(colorPropertyTransformed)
			colorPropertyTransformed /= np.max(colorPropertyTransformed)
		
			graph.vs["color"] = [convertColorToHex(*colormap(value)) for value in colorPropertyTransformed]
		else:
			communities = graph.vs["Community"];
			sortedCommunities = sortByFrequency(communities);
			communityToColor = {community:(_styleColors[index] if index<len(_styleColors) else "#aaaaaa") for index,community in enumerate(sortedCommunities)};
			graph.vs["color"] = [communityToColor[community] for community in communities];
		
		for edgeIndex in range(graph.ecount()):
			sourceIndex = graph.es[edgeIndex].source
			graph.es[edgeIndex]['color'] = graph.vs["color"][sourceIndex]+"20"

		outputBaseName,outputExtension = os.path.splitext(filename)
		outputFile = os.path.join(figuresOutputDirectory,outputBaseName+".pdf")
		fig, ax = plt.subplots(figsize=(10,10))
		if(plotLabels):
			g.vs["name"] = labels;
		drawGraph(graph,ax)
		plt.axis("off")
		ax.set_facecolor("grey")
		plt.savefig(outputFile)
		plt.close()
		
		
		# with open(os.path.join(outputDirectory,os.path.basename(filename)), "w") as fd:
		# 	if(weighted):
		# 		outputData = g.get_adjacency(attribute='weight').data
		# 	else:
		# 		outputData = g.get_adjacency().data
		# 	np.savetxt(fd,outputData,delimiter=",")

with open(os.path.join(outputDirectory,"index.json"), "w") as fd:
	json.dump(indexData,fd)

with open(os.path.join(outputDirectory,"label.json"), "w") as fd:
	json.dump(labelData,fd)

