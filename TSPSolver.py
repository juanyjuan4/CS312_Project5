#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import uuid



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self.bssf = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	# Create a matrix of distances, and make it reduced-cost if needed.
	def createMatrix(self, reduce):
		lowerBound = 0
		ncities = len(self._scenario._cities)
		matrix = np.empty((ncities, ncities))
		matrix.fill(np.inf)
		
		# Add the path distances into the matrix.
		for i in range(0, ncities):
			for j in range(0, ncities):
				if self._scenario._edge_exists[i, j]:
					matrix[i, j] = self._scenario._cities[i].costTo(self._scenario._cities[j])
		
		# Reduce the matrix if needed.
		if reduce:
			rowMins = np.amin(matrix, axis = 1)
			for i in range(0, ncities):
				matrix[i, :] -= rowMins[i]
				lowerBound += rowMins[i]
			
			colMins = np.amin(matrix, axis = 0)
			for i in range(0, ncities):
				if colMins[i] > 0:
					matrix[:, i] -= colMins[i]
					lowerBound += colMins[i]
		
		return matrix, lowerBound
		

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	# This is used as my initial BSSF.
	def greedy( self,time_allowance=60.0 ):
		start_time = time.time()
		results = {}
		ncities = len(self._scenario._cities)
		matrix, lowerBound = self.createMatrix(False)
		total = 1
		
		currentIndex = 0
		path = []
		path.append(self._scenario._cities[currentIndex])
		# While not all cities have been visited, this algorithm finds the next shortest path,
		# and sets unavailable paths to infinity.
		while len(path) < ncities:
			matrix[currentIndex, 0] = np.inf
			nextMin = np.amin(matrix[currentIndex, :])
			lowerBound += nextMin
			nextIndex = np.where(matrix[currentIndex, :] == nextMin)[0][0]
			path.append(self._scenario._cities[nextIndex])
			matrix[:, nextIndex] += np.repeat(np.inf, ncities)
			matrix[currentIndex, :] += np.repeat(np.inf, ncities)
			currentIndex = nextIndex
			total += 1
		
		if matrix[currentIndex, 0] == np.inf: # Check if the path is viable or not.
			self.bssf = TSPSolution([self._scenario._cities[0], self._scenario._cities[0]])
		else:
			self.bssf = TSPSolution(path)
		
		end_time = time.time()
		
		results['cost'] = self.bssf.cost
		results['time'] = end_time - start_time
		results['count'] = None
		results['soln'] = self.bssf
		results['max'] = None
		results['total'] = total
		results['pruned'] = None
		return results



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		start_time = time.time()
		results = {}
		ncities = len(self._scenario._cities)
		matrix, lowerBound = self.createMatrix(True)
		count = 0
		pruned = 0
		total = 1
		
		self.greedy() # Run the greedy algorithm to fill self.bssf for use later.
		
		heap = []
		hashMap = {} # Used to store other data from the priority queue that I didn't want messing up the ordering.
		maxHeapSize = 0
		currentIndex = 0
		path = []
		path.append(self._scenario._cities[currentIndex])
		for i in range(1, ncities):
			total += 1
			tempBound = lowerBound + matrix[currentIndex, i]
			tempMatrix = matrix.copy()
			tempMatrix[:, i] += np.repeat(np.inf, ncities)
			tempMatrix[currentIndex, :] += np.repeat(np.inf, ncities)
			rowMins = np.amin(tempMatrix, axis = 1)
			tempPath = path + [self._scenario._cities[i]]
   
			for j in range(0, ncities):
				if rowMins[j] == np.inf:
					continue
				tempMatrix[j, :] -= np.repeat(rowMins[j], ncities)
				tempBound += rowMins[j]
    
			if tempBound < self.bssf.cost:
				tempID = str(uuid.uuid1()) # Create an ID for hashing data.
				hashMap[tempID] = (i, tempMatrix.copy(), tempPath) # Add important data to hashmap.
				heapq.heappush(heap, (tempBound, tempID)) # Add current path cost to queue, with associated hash.
				if maxHeapSize < len(heap):
					maxHeapSize = len(heap)
			else:
				pruned += 1
		
		# Expand values from the queue until best path found or time runs out.
		# This code is very similar to the block above.
		while heap and time.time() - start_time < time_allowance:
			currentBound, currentID = heapq.heappop(heap) # Pop off the queue.
			currentIndex, currentMatrix, currentPath = hashMap[currentID] # Retrieve data from the hash table.
			if currentBound > self.bssf.cost:
				pruned += 1
				continue
   
			for i in range(0, ncities):
				if i == currentIndex: # No need to check paths to itself.
					continue
				if i == 0 and len(currentPath) != ncities: # If it's not at the end, don't check the first location.
					continue
				total += 1
				tempBound = currentBound + currentMatrix[currentIndex, i]
				tempMatrix = currentMatrix.copy()
				# Infinite out blocked paths, and find the minimum values from each row.
				tempMatrix[:, i] += np.repeat(np.inf, ncities)
				tempMatrix[currentIndex, :] += np.repeat(np.inf, ncities)
				rowMins = np.amin(tempMatrix, axis = 1)
				# Add to the path.
				tempPath = currentPath + [self._scenario._cities[i]]
    
				for j in range(0, ncities):
					if rowMins[j] == np.inf:
						continue
					tempMatrix[j, :] -= np.repeat(rowMins[j], ncities)
					tempBound += rowMins[j]
     
				if tempBound < self.bssf.cost:
					if len(tempPath) == ncities + 1 and i == 0: # See if a full path was made, and then update BSSF.
						del tempPath[-1] # Take off the last value since it's a repeat.
						self.bssf = TSPSolution(tempPath)
						count += 1
					else:
						tempID = str(uuid.uuid1()) # Add to the queue.
						hashMap[tempID] = (i, tempMatrix.copy(), tempPath)
						heapq.heappush(heap, (tempBound, tempID))
						if maxHeapSize < len(heap):
							maxHeapSize = len(heap)
				else:
					pruned += 1
		
		end_time = time.time()
  
		while heap:
			bound, ID = heapq.heappop(heap)
			if bound > self.bssf.cost:
				pruned += 1
			
		results['cost'] = self.bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = self.bssf
		results['max'] = maxHeapSize
		results['total'] = total
		results['pruned'] = pruned
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass
