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



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

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
		
		for i in range(0, ncities):
			for j in range(0, ncities):
				if self._scenario._edge_exists[i, j]:
					matrix[i, j] = self.scenario._cities[i].costTo(self.scenarios._cities[j])
		
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

	def greedy( self,time_allowance=60.0 ):
		results = {}
		ncities = len(self._scenario._cities
		matrix, lowerBound = self.createMatrix(False)
		
		currentIndex = 0
		path = [self._scenarios._cities[currentIndex]]
		while len(path) < ncities + 1:
			nextMin = np.amin(matrix[currentIndex, :])
			lowerBound += nextMin
			nextIndex = np.where(matrix[currentIndex, :] == nextMin)
			path.append(self._scenario._cities[nextIndex])
			matrix[:, nextIndex] += np.inf
			currentIndex = nextIndex
		
		self.bssf = TSPSolution(path)
		
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
		results = {}
		ncities = len(self._scenario._cities)
		matrix, lowerBound = self.createMatrix(True)
		
		self.greedy()
		
		heap = []
		currentIndex = 0
		path = [self._scenarios._cities[currentIndex]]
		for i in range(1, ncities):
			tempBound = lowerBound + matrix[currentIndex, i]
			tempMatrix = matrix
			tempMatrix[:, i] += np.inf
			rowMins = np.amin(tempMatrix, axis = 1)
			tempPath = path + [self._scenarios._cities[i]]
			for j in range(0, ncities):
				tempMatrix[j, :] -= rowMins[j]
				tempBound += rowMins[j]
			if tempBound < self.bssf._costOfRoute():
				heapq.heappush(heap, (tempBound, i, tempMatrix, tempPath))
		
		while heap:
			currentBound, currentIndex, currentMatrix, currentPath = heapq.heappop(heap)
			for i in range(0, ncities):
				if i == index:
					continue
				tempBound = currentBound + currentMatrix[currentIndex, i]
				tempMatrix = currentMatrix
				tempMatrix[:, i] += np.inf
				rowMins = np.amin(tempMatrix, axis = 1)
				tempPath = currentPath + [self._scenarios._cities[i]]
				for j in range(0, ncities):
					tempMatrix[j, :] -= rowMins[j]
					tempBound += rowMins[j]
				if tempBound < self.bssf._costOfRoute():
					if i == 0:
						self.bssf = TSPSolution(tempPath)
					else:
						heapq.heappush(heap, (tempBound, i, tempMatrix, tempPath))
				
		
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