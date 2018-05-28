def main():
    # import numpy as np
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    # from datetime import datetime


    alldata = open('./processed.data')
    processed = []
    training = []
    testing = []
    startpoints = [-1,-1,-1,-1,-1]
    training_labels = []
    testing_labels = []

    # Cleans up the data, removing any instances that have missing data
    for line in alldata:
        temp = []
        if "?" in line:
            continue
        for val in line.split(','):
            temp.append(float(val.strip()))
        processed.append(temp)
    alldata.close

    # Initially, each entry has its label in the last column. First we will sort by this value (0-4)
    processed = sorted(processed, key=lambda x : x[-1])

    # Here, we get the first instance of each value
    linecount = 0
    for line in processed:
        lastval = int(line[-1])
        if startpoints[lastval] == -1:
            startpoints[lastval] = linecount
        linecount += 1


    # Here we get the training and testing data sets by choosing the first 60% of the total data set, floored.
    # Whatever remains goes into the testing data set.
    numCount = 0
    for start in startpoints:
        count = getSplit(countDistribution(processed))[numCount]
        total = count[0] + count[1]
        for x in range(total):
            if x < count[0]:
                training.append(processed[start+x])
            else:
                testing.append(processed[start+x])
        numCount += 1

    # We now remove the columns that contain the final diagnosis, and put them in the label lists.
    for vals in training:
        label = vals[-1]
        training_labels.append(label)
        vals = vals[0:len(vals)-1]

    for vals in testing:
        label = vals[-1]
        testing_labels.append(label)
        vals = vals[0:len(vals)-1]

    # We use sklearn's StandardScaler to scale our data
    scaler = StandardScaler() 
    scaler.fit(training)
    training = scaler.transform(training)
    testing = scaler.transform(testing)

    # 
    svmClassifier = svm.SVC()
    svmClassifier.fit(training, training_labels)
    print svmClassifier.score(testing, testing_labels)
    print svmClassifier.score(training, training_labels)
    results = svmClassifier.predict(training)
    getFalseNegatives(results,training_labels)


def countDistribution(myData):
    zerocount = 0
    onecount = 0
    twocount = 0
    threecount = 0
    for x in myData:
        if x[-1] == 0.0:
            zerocount+= 1
        elif x[-1] == 1.0:
            onecount += 1
        elif x[-1] == 2.0:
            twocount += 1
        elif x[-1] == 3.0:
            threecount += 1
    fourcount = len(myData) - zerocount - onecount - twocount - threecount
    return [zerocount, onecount, twocount, threecount, fourcount]
    
def getSplit(distribution):
    ranges = []
    for vals in distribution:
        sixty = int(vals*0.6)
        forty = vals - sixty
        ranges.append([sixty,forty])
    return ranges

def getFalseNegatives(results, actual):
    for x in range(len(results)):
        print results[x], "???", actual[x]
    pass

if __name__ == "__main__":
    main()