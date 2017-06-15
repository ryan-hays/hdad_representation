#input: a sparse representation. list of tuples where tuple[0] is (x, y, z) coord, and tuple[1] is atom type ("C", "N", "O", etc.)
import math
import peakutils
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pairwise
import sklearn.metrics.regression as regressionScore 
import sklearn.kernel_ridge as kernel_ridge
import itertools

def distance_dict(atom_list):
	"""
	Returns a dictionary mapping every atom pair to a list of associated distances
	"""
	distanceDict = {}
	for atom in atom_list:
		coord = atom[0]
		atomType = atom[1]
		distanceInfo = distance_generation(atom_list, coord, atomType)
		for pair in distanceInfo:
			if pair[0] not in distanceDict:
				distanceDict[pair[0]] = [pair[1]]
			else:
				distanceDict[pair[0]].append(pair[1])
	return distanceDict


def angle_dict(atom_list):
	"""
	Returns a dict mapping atom triples to angle values
	"""
	angleDict = {}
	NUMBER_NEIGHBORS = 3
	for atom in atom_list:
		neighbors = find_nearest_neighbors(atom, atom_list, NUMBER_NEIGHBORS)
		#Create subset permutations
		subsets = [[atom, neighbors[0], neighbors[1]], [atom, neighbors[0], neighbors[2]], [atom, neighbors[1], neighbors[2]]]
		for s in subsets:			
			label = tuple([s[0][1]]+sorted([info[1] for info in s[1:]]))
			#generate angle
			vec1 = np.array(s[1][0])-np.array(s[0][0])
			vec2 = np.array(s[2][0])-np.array(s[0][0])
			angle = math.acos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
			if label not in angleDict:
				angleDict[label] = [angle]
			else:
				angleDict[label].append(angle)
	return angleDict


def dihedral_angle_dict(atom_list):
	"""
	Returns a dict mapping atom quads to dihedral angles values
	"""
	NUMBER_NEIGHBORS = 4
	diAngleDict = {}
	for atom in atom_list:
		neighbors = find_nearest_neighbors(atom, atom_list, 4)		
		subsets = list(itertools.combinations(neighbors, 3))		
		for sub in subsets:
			lowerSubsets = list(itertools.combinations(list(sub), 2))
			#Find neighbors not in subsets
			for neigh in neighbors:
				if neigh not in sub:
					secondAtom = neigh 
			for low in lowerSubsets:
				label = tuple([atom[1]]+sorted([secondAtom[1]]+[l[1] for l in low]))
				planeOnePoints = [np.array(atom[0]), np.array(secondAtom[0]), np.array(low[0][0])]
				planeTwoPoints = [np.array(atom[0]), np.array(secondAtom[0]), np.array(low[1][0])]
				planeOne = find_plane_equation(planeOnePoints)
				planeTwo = find_plane_equation(planeTwoPoints)
				try:
					diAngle = math.acos(np.dot(planeOne, planeTwo)/(np.linalg.norm(planeOne)*np.linalg.norm(planeTwo)))
				except ValueError:
					pass
				if label not in diAngleDict:
					diAngleDict[label] = [diAngle]
				else:
					diAngleDict[label].append(diAngle)
	return diAngleDict


####### Helper Functions #############


def find_nearest_neighbors(atom, atom_list, numNeighbs):
	distanceList = [] #populated with (coordinate, label, distance)	
	for neighbor in atom_list:
		if neighbor != atom:
			#Calculate distance 
			distance = 0
			for i in range(len(atom[0])):
				distance += (atom[0][i]-neighbor[0][i])**2
			distance = distance**0.5
			distanceList.append([neighbor[0], neighbor[1], distance])
	sortDist = sorted(distanceList, key=lambda val: val[2])[:numNeighbs]
	noDist = [entry[:2] for entry in sortDist] #remove distances from entries	
	return noDist #find 3 closest neighbors #ERROR


def distance_generation(atom_list, coordinate, atomType):
	result = []
	for atom in atom_list:
		if atom[0] != coordinate:
			atomPair = tuple(sorted([atomType, atom[1]]))
			distance = 0
			for i in range(len(atom[0])):
				distance += (atom[0][i]-coordinate[i])**2
			distance = distance**0.5
			pairInfo = (atomPair, distance)
			result.append(pairInfo)
	return result

def find_plane_equation(points):
	"""
	Given three points, return plane going through all of them
	"""
	p1 = points[0]
	p2 = points[1]
	p3 = points[2]
	# These two vectors are in the plane
	v1 = p3 - p1
	v2 = p2 - p1
	# the cross product is a vector normal to the plane
	cp = np.cross(v1, v2)
	a, b, c = cp
	# This evaluates a * x3 + b * y3 + c * z3 which equals d
	d = np.dot(cp, p3)
	return np.array([a, b, c])

def is_close(a, b, rel_tol=1e-2, abs_tol=0.0):
	return abs(a-b) <= max( rel_tol * max(abs(a), abs(b)), abs_tol )

def uniform_array(array):
	uniVal = array[0]
	for val in array[1:]:
		if val != uniVal:
			return False 
	return True

def generate_interact_frequencies(dataPoints):
	#Round values to 
	frequencyDict = {}
	for point in dataPoints:
		foundVal = False
		for val in frequencyDict:
			if is_close(point, val):
				frequencyDict[val] += 1
				foundVal = True
				break
		if not foundVal:
			frequencyDict[point] = 1
	xVals = sorted(frequencyDict.keys())
	yVals = [frequencyDict[x] for x in xVals]
	return (xVals, yVals)

def find_bin_locations(label, dataPoints):
	"""
	label: (string) represents the information atom type
	dataPoints: (list) contains all floating point values of given label

	returns: a tuple with (label, bins_locations)
	"""
	xVals, yVals = generate_interact_frequencies(dataPoints)
	#Check for uniform frequencies
	if uniform_array(yVals):
		peakIndexes = np.array([int(math.floor(len(xVals)/2))])

	else:
		peakIndexes = peakutils.indexes(np.array(yVals))
		#Pick random ones if too big
		if len(peakIndexes) > 25:
			peakIndexes = np.random.choice(peakIndexes, 25, replace=False)
	localExtrema = [xVals[i] for i in peakIndexes.tolist()]
	return (label, localExtrema)

def interpolate_sample(localExtrema, dataPoints):
	featureVals = [] #histogram repersented as dictionary {bin: binVal}
	for i in range(len(localExtrema)): #iterate over bins using index vals
		runningSum = 0
		if len(localExtrema) > 1: 
			if i == 0: #first bin, doesn't have neighbor before 
				for point in dataPoints:
					runningSum += max(0, (point-localExtrema[i+1])/(localExtrema[i]-localExtrema[i+1]))
			elif i == len(localExtrema) - 1: #then at the end, only neighbor is in+-
				for point in dataPoints:
					runningSum += max(0, (point-localExtrema[i-1])/(localExtrema[i]-localExtrema[i-1]))
			else:
				for point in dataPoints:
					runningSum += max(0, min((point-localExtrema[i+1])/(localExtrema[i]-localExtrema[i+1]), (point-localExtrema[i-1])/(localExtrema[i]-localExtrema[i-1])))
			featureVals.append(runningSum)
		else: #just 1 bin
			for point in dataPoints:
				runningSum += max(0, (point-localExtrema[i])/localExtrema[i])
			featureVals.append(runningSum)
	return featureVals

def parse_molecule(filename, directory):
	molecule = np.load(directory+'/'+filename)
	atomList = []
	for i in range(molecule.shape[0]):
		atomList.append([molecule[i][:3].tolist()]+[backwards_atom_dict(molecule[i][3])])
	distanceDict = distance_dict(atomList)
	angleDict = angle_dict(atomList)
	diAngleDict = dihedral_angle_dict(atomList)
	return (distanceDict, angleDict, diAngleDict)

def dictionary_build(mainDict, newDict):
	for item in newDict:
		if item in mainDict:
			mainDict[item] += newDict[item]
		else:
			mainDict[item] = newDict[item]

def backwards_atom_dict(elementInt):
	backAtomDict = {0.0:'H', 5.0:'B', 6.0:'C', 7.0:'N', 8.0:'O', 9.0:'F', 7.2:'P', 8.2:'S', 9.2:'Cl', 12:'V', 8.4:'Se', 9.4:'Br', 9.6:'Mg', 9.5:'Zn', 9.7:'Fe'}
	try:
		return backAtomDict[elementInt]
	except KeyError:
		return 'H'

def feature_vector_create(directory, interactMap, binLocationDict):
	sampleToFeatureVector = {}
	sampleToData = {}
	parsedMolecules = []
	allInteractions = {} 
	for filename in os.listdir(directory):
		base, ext = os.path.splitext(filename)
		if ext == '.npy': 
			parseData = parse_molecule(filename, directory)
			parsedMolecules.append(parseData)
			sampleToData[filename] = parseData
			sampleToFeatureVector[filename] = []
	print(directory)
	#Combine all sample data into 1 dictionary to find bin locations
	for mol in parsedMolecules:
		for obj in mol:
			dictionary_build(allInteractions, obj)
	#Now find bin locations for each interaction
	for i in range(len(interactMap.keys())):
		interact = interactMap[i]
		binLocations = binLocationDict[interact]
		#Now interpolate for each molecule, and add to their feature vectors
		for mol in sampleToData:
			foundInteract = False
			for dictType in sampleToData[mol]:
				if interact in dictType:
					interpolatedData = interpolate_sample(binLocations, dictType[interact])
					sampleToFeatureVector[mol] += interpolatedData
					foundInteract = True
					break
			if not foundInteract: 
				sampleToFeatureVector[mol] += [0.0]*len(binLocations)
	#Now concatenate all feature vectors
	valueSet = np.array(sampleToFeatureVector.values())
	np.save(directory+".npy", valueSet)
	return valueSet

def representation_build(directory):
	sampleToFeatureVector = {}
	sampleToData = {}
	parsedMolecules = []
	allInteractions = {} 
	for filename in os.listdir(directory):
		base, ext = os.path.splitext(filename)
		if ext == '.npy': 
			parseData = parse_molecule(filename, directory)
			parsedMolecules.append(parseData)
			sampleToData[filename] = parseData
			sampleToFeatureVector[filename] = []
	print(directory)
	#Combine all sample data into 1 dictionary to find bin locations
	for mol in parsedMolecules:
		for obj in mol:
			dictionary_build(allInteractions, obj)
	#Map interaction type to feature vector index
	interactMap = {}
	i = 0
	for interact in allInteractions:
		interactMap[i] = interact
		i += 1
	#Now find bin locations for each interaction
	binLocationDict = {}
	for i in range(len(interactMap.keys())):
		interact = interactMap[i]
		bins = find_bin_locations(interact, allInteractions[interact])
		binLocationDict[bins[0]] = bins[1]
		#Now interpolate for each molecule, and add to their feature vectors
		for mol in sampleToData:
			foundInteract = False
			for dictType in sampleToData[mol]:
				if interact in dictType:
					interpolatedData = interpolate_sample(bins[1], dictType[interact])
					sampleToFeatureVector[mol] += interpolatedData
					foundInteract = True
					break
			if not foundInteract: 
				sampleToFeatureVector[mol] += [0.0]*len(bins[1])
	#Now concatenate all feature vectors
	trainingSet = np.array(sampleToFeatureVector.values())
	np.save('databaseBuild.npy', trainingSet)
	return (trainingSet, interactMap, binLocationDict)

if __name__ == '__main__':
	featureExtract = representation_build('QM9database')
	interactions = featureExtract[1]
	featureLocations = featureExtract[2]
	np.save('featureLocations.npy', featureLocations)
	trainingSet = feature_vector_create('QM9TrainingNP', interactions, featureLocations)
	validationSet = feature_vector_create('QM9ValidationNP', interactions, featureLocations)
	#Regression
	regress = kernel_ridge.KernelRidge(alpha=1, kernel='rbf', gamma=1e-9)
	trainingVector = np.load('trainingData.npy')
	validationVector = np.load('validationData.npy')
	regress.fit(trainingSet, trainingVector)
	trainingPredict = regress.predict(trainingSet)
	validationPredict = regress.predict(validationSet)
	trainingError = regressionScore.mean_squared_error(trainingVector, trainingPredict)
	validationError = regressionScore.mean_squared_error(validationVector, validationPredict)
	print("Training MSE: ", trainingError)
	print("Validation MSE: ", validationError)



	# #TEST: plot histograms for one
	# distance = plt.figure(1) #first figure is C00
	# angle = plt.figure(2)
	# dihedralAngle = plt.figure(3)
	# plt.figure(1)
	# plt.hist(allInteractions[('C', 'N')], bins=5000, log=True)
	# plt.title('C-N Distances')
	# plt.xlabel('Angstroms')
	# plt.ylabel('Frequency')
	# distance.savefig('CNwith5000binsLog.png')
	# plt.figure(2)
	# plt.hist(allInteractions[('C', 'C', 'C')], bins=5000, log=True)
	# plt.title('C-C-C Angles')
	# plt.xlabel('Radians')
	# plt.ylabel('Frequency')
	# angle.savefig('CCCwith5000binsLog.png')
	# plt.figure(3)
	# plt.hist(allInteractions[('C', 'C', 'C', 'O')], bins=5000, log=True)
	# plt.title('C-C-C-O Dihedral Angles')
	# plt.xlabel('Radians')
	# plt.ylabel('Frequency')
	# dihedralAngle.savefig('CCOOwith5000binsLog.png')
	# carbonDist = plt.figure(4)
	# plt.figure(4)
	# plt.hist(allInteractions[('C', 'C')], bins=5000, log=True)
	# plt.title('C-C Distances')
	# plt.xlabel('Angstroms')
	# plt.ylabel('Frequency')
	# carbonDist.savefig('carbonDist.png')

	






