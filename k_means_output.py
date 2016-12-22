from __future__ import print_function

# $example on$
from numpy import array
from math import sqrt
# $example off$

from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
# $example off$



if __name__ == "__main__":
	sc = SparkContext(appName="KMeansOutput")  # SparkContext

	# $example on$
	# Load and parse the data
	data = sc.textFile("kmeans_data.txt")
	parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

	# Build the model (cluster the data)
	clusters = KMeans.train(parsedData, 4, maxIterations=100, runs=10, initializationMode="random")

	lines = clusters.predict(parsedData)

	#lines = parsedData.map(lambda line: array([int(x) for x in line[0]]))
	#lines = parsedData.map(lambda line: array([clusters.predict(x) for x in parsedData]))

	#lines = np.array([])
	#i = 1
	#for x in parsedData:
	#	k = clusters.predict(x)
	#	np.append(lines,[i,k])
	#	i += 1
	#def toCSVLine(data):
	#	return ' '.join(str(d) for d in data)

	#lines = lines.map(toCSVLine)
	lines = lines.zipWithIndex()
	#lines = lines.map(lambda line: array([i, x for (i,x) in zip(range(len(line)),line)]))
	lines.repartition(1).saveAsTextFile('/Users/Lynn/Desktop/spark-2.0.2-bin-hadoop2.7/kmeans_output.csv')
	#coalesce(1,True)

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    # def error(point):
    #     center = clusters.centers[clusters.predict(point)]
    #     return sqrt(sum([x**2 for x in (point - center)]))

    # WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    # print("Within Set Sum of Squared Error = " + str(WSSSE))

    # # Save and load model
    # clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
    # sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
    # # $example off$

	sc.stop()
