
"""
Collaborative Filtering Classification Example.
"""
from __future__ import print_function

from pyspark import SparkContext

# $example on$
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
# $example off$
import csv


if __name__ == "__main__":
    sc = SparkContext(appName="PythonCollaborativeFiltering")
    # $example on$
    # Load and parse the data

    numIterations = 15
    rank = 150

    data = sc.textFile("0.txt")
    ratings = data.map(lambda l: l.split(' '))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    model = ALS.train(ratings, rank, numIterations)
    result = model.recommendProductsForUsers(7)


    for num in range(1,100):
        filename = str(num)+".txt"
        data = sc.textFile(filename)
        ratings = data.map(lambda l: l.split(' '))\
            .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
        model = ALS.train(ratings, rank, numIterations)
        result = result.union(model.recommendProductsForUsers(7))


    cf_result = result.collect()

    myfile = open('cf_result.csv','w')
    wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
    for i in range(len(cf_result)):
        wr.writerow((cf_result[i][0],cf_result[i][1][0][1],cf_result[i][1][1][1],cf_result[i][1][2][1],cf_result[i][1][3][1],cf_result[i][1][4][1],cf_result[i][1][5][1],cf_result[i][1][6][1]))


