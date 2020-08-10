#!/usr/bin/env python

import sys
import os.path
from os.path import join as PJ
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
import base64
import io 
import jgf

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

def isFloat(value):
	if(value is None):
		return False
	try:
		numericValue = float(value)
		return np.isfinite(numericValue)
	except ValueError:
		return False


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
			np.int16, np.int32, np.int64, np.uint8,
			np.uint16, np.uint32, np.uint64)):
			ret = int(obj)
		elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
			ret = float(obj)
		elif isinstance(obj, (np.ndarray,)): 
			ret = obj.tolist()
		else:
			ret = json.JSONEncoder.default(self, obj)

		if isinstance(ret, (float)):
			if math.isnan(ret):
				ret = None

		if isinstance(ret, (bytes, bytearray)):
			ret = ret.decode("utf-8")

		return ret

results = {"errors": [], "warnings": [], "brainlife": [], "datatype_tags": [], "tags": []}

def warning(msg):
	global results
	results['warnings'].append(msg) 
	#results['brainlife'].append({"type": "warning", "msg": msg}) 
	print(msg)

def error(msg):
	global results
	results['errors'].append(msg) 
	#results['brainlife'].append({"type": "error", "msg": msg}) 
	print(msg)

def exitApp():
	global results
	with open("product.json", "w") as fp:
		json.dump(results, fp, cls=NumpyEncoder)
	if len(results["errors"]) > 0:
		sys.exit(1)
	else:
		sys.exit()

def exitAppWithError(msg):
	global results
	results['errors'].append(msg) 
	#results['brainlife'].append({"type": "error", "msg": msg}) 
	print(msg)
	exitApp()







configFilename = "config.json"
argCount = len(sys.argv)
if(argCount > 1):
		configFilename = sys.argv[1]

outputDirectory = "output"

figuresOutputDirectory = os.path.join(outputDirectory, "figures")

outputFile = PJ(outputDirectory,"network.json.gz")

if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
		config = json.load(fd)


if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)


# "color-property": "degree",
# "size-property":"degree"

colorProperty = "Degree"
sizeProperty = "Degree"
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
	positions = np.array(graph.layout_lgl(maxiter=1000,coolexp = 2.0))
	# print("Plotting...");
	linesX = []
	linesY = []
	segments = []
	positionsX = positions[:,0]
	positionsY = positions[:,1]
	lineWidths = []
	isWeighted = False
	if("weight" in graph.edge_attributes()):
		isWeighted= True
		maxWeight = np.max(graph.es["weight"])
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

		width = 1.5
		if(isWeighted):
			width = edge["weight"]/maxWeight*4+0.5
		lineWidths.append(width)
		# lineWidths.append(0.0)
		# lineWidths.append(width)

		segments.append([(fx, fy), (tx, ty)])
	# plt.plot(linesX,linesY,alpha=0.1);
	lc = mc.LineCollection(segments, colors=graph.es["color"], linewidths=lineWidths)
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
	

networks = jgf.igraph.load(config["network"],compressed=True)

#Online first network is plotted
if(len(networks)>0):
	network = networks[0]
	weighted = "weight" in network.edge_attributes()
	hasCommunities = "Community" in network.vertex_attributes()
	
	if("degree" not in network.vertex_attributes()):
		network.vs["degree"] = network.degree()
	if("inDegree" not in network.vertex_attributes()):
		network.vs["indegree"] = network.indegree()
	if("outDegree" not in network.vertex_attributes()):
		network.vs["outdegree"] = network.outdegree()
	if("strength" not in network.vertex_attributes()):
		network.vs["strength"] = network.strength(weights = ('weight' if weighted else None))
	if("instrength" not in network.vertex_attributes()):
		network.vs["instrength"] = network.strength(mode="IN", weights = ('weight' if weighted else None))
	if("outstrength" not in network.vertex_attributes()):
		network.vs["outstrength"] = network.strength(mode="OUT", weights = ('weight' if weighted else None))
		
	print(sizeProperty)
	sizeArray = network.vs[sizeProperty]
	maxProperty = max(sizeArray)

	network.vs["vertex_size"] = [x/maxProperty*200+4 for x in sizeArray]
	
	
	if("Community" not in network.vertex_attributes() or colorProperty!="community"):
		colormap = plt.get_cmap("winter")
		if(colorProperty not in network.vertex_attributes()):
			colorProperty="degree"
		colorPropertyArray = np.array(network.vs[colorProperty])
		colorPropertyTransformed = np.log(colorPropertyArray+1.0)
		colorPropertyTransformed -= np.min(colorPropertyTransformed)
		colorPropertyTransformed /= np.max(colorPropertyTransformed)
	
		network.vs["color"] = [convertColorToHex(*colormap(value)) for value in colorPropertyTransformed]
	else:
		communities = network.vs["Community"];
		sortedCommunities = sortByFrequency(communities);
		communityToColor = {community:(_styleColors[index] if index<len(_styleColors) else "#aaaaaa") for index,community in enumerate(sortedCommunities)};
		network.vs["color"] = [communityToColor[community] for community in communities];
	
	for edgeIndex in range(network.ecount()):
		sourceIndex = network.es[edgeIndex].source
		network.es[edgeIndex]['color'] = network.vs["color"][sourceIndex]+"20"

	outputFile = os.path.join(outputDirectory,"report.pdf")
	fig, ax = plt.subplots(figsize=(10,10))
	
	drawGraph(network,ax)
	plt.axis("off")
	ax.set_facecolor("grey")
	plt.savefig(outputFile)
	pic_IObytes = io.BytesIO()
	plt.savefig(pic_IObytes,dpi=35,  format='png')
	pic_IObytes.seek(0)
	pic_hash = base64.b64encode(pic_IObytes.read())
	results["brainlife"].append( { 
			"type": "image/png", 
			"name": "Visualization",
			"base64": pic_hash.decode("utf8"),
	})
	plt.close()
		
exitApp()








# 		if(not isNullModel):
# 			finalNodeMeasurements[aggregateName] = nodeProperties 
# 			finalNetworkMeasurements[aggregateName] = networkProperties
# 		else:
# 			if(aggregateName not in nullmodelNodeMeasurements):
# 				nullmodelNodeMeasurements[aggregateName] = {}
# 				nullmodelNetworkMeasurements[aggregateName] = {}
# 			for measurement,propData in nodeProperties.items():
# 				if(measurement not in nullmodelNodeMeasurements[aggregateName]):
# 					nullmodelNodeMeasurements[aggregateName][measurement] = []
# 				nullmodelNodeMeasurements[aggregateName][measurement].append(propData)
# 			for measurement,propData in networkProperties.items():
# 				if(measurement not in nullmodelNetworkMeasurements[aggregateName]):
# 					nullmodelNetworkMeasurements[aggregateName][measurement] = []
# 				nullmodelNetworkMeasurements[aggregateName][measurement].append(propData)

		
# 		outputBaseName,outputExtension = os.path.splitext(filename)

# 		if("Community" in g.vertex_attributes()):
# 			with open(PJ(csvOutputDirectory,"%s_community.txt"%os.path.basename(outputBaseName)), "w") as fd:
# 				for item in g.vs["Community"]:
# 					fd.write("%s\n"%str(item))
		
# 		for nodeProperty,nodePropData in nodeProperties.items():
# 			propFilename = PJ(csvOutputDirectory,"%s_prop_%s.txt"%(os.path.basename(inputBaseName),nodeProperty))
# 			np.savetxt(propFilename,nodePropData);

# 		if("properties" not in entry):
# 			entry["properties"] = list(nodeProperties.keys());

# 		with open(PJ(csvOutputDirectory,os.path.basename(filename)), "w") as fd:
# 			if(weighted):
# 				outputData = g.get_adjacency(attribute='weight').data
# 			else:
# 				outputData = g.get_adjacency().data
# 			np.savetxt(fd,outputData,delimiter=",")

# 	# print(finalNetworkMeasurements);
# 	# nullmodelNodeMeasurements = {};

# 	# print(nullmodelNetworkMeasurements);
# 	# nullmodelNetworkMeasurements = {};
# 	for aggregatorName in finalNetworkMeasurements:
# 		with open(PJ(csvOutputDirectory,"%s__measurements.csv"%aggregatorName), "w") as fd:
# 			fd.write("Measurement,Value,NullModels\n")
# 			for measurement,value in finalNetworkMeasurements[aggregatorName].items():
# 				fd.write("%s,%0.18g"%(measurement,value))
# 				if(aggregatorName in nullmodelNetworkMeasurements
# 					and measurement in nullmodelNetworkMeasurements[aggregatorName]):
# 					nullValues = nullmodelNetworkMeasurements[aggregatorName][measurement]
# 					fd.write(","+",".join(["%0.18g"%nullValue for nullValue in nullValues]))
# 					_,bins = np.histogram([value]+nullValues,bins=30)
# 					if(shallPlot):
# 						fig = plt.figure(figsize= (8,5))
# 						ax = plt.axes()
# 						ax.hist(nullValues,bins=bins,density=True,color="#888888")
# 						ax.hist([value],bins=bins,density=True,color="#cc1111")
# 						ax.set_xlabel(measurement);
# 						ax.set_ylabel("Density");
# 						fig.savefig(PJ(figuresOutputDirectory,"network_hist_%s_%s.pdf"%(aggregatorName,measurement)));
# 						plt.close(fig)
# 				fd.write("\n")
	
# 	if(shallPlot):
# 		for aggregatorName in finalNodeMeasurements:
# 			for measurement,values in finalNodeMeasurements[aggregatorName].items():
# 					nullValues = [];
# 					if(aggregatorName in nullmodelNodeMeasurements
# 						and measurement in nullmodelNodeMeasurements[aggregatorName]):
# 						nullValues = list(np.array(nullmodelNodeMeasurements[aggregatorName][measurement]).flatten())
# 					_,bins = np.histogram(list(values)+nullValues,bins=30)
# 					fig = plt.figure(figsize= (8,5))
# 					ax = plt.axes()
# 					if(nullValues):
# 						ax.hist(nullValues,bins=bins,density=True,color="#888888")
# 					ax.hist(values,bins=bins,density=True,color="#cc1111",alpha=0.75)
# 					ax.set_xlabel(measurement);
# 					ax.set_ylabel("Density");
# 					fig.savefig(PJ(figuresOutputDirectory,"nodes_hist_%s_%s.pdf"%(aggregatorName,measurement)));
# 					plt.close(fig)
		


# with open(PJ(outputDirectory,"index.json"), "w") as fd:
# 	json.dump(indexData,fd)

# with open(PJ(outputDirectory,"label.json"), "w") as fd:
# 	json.dump(labelData,fd)

