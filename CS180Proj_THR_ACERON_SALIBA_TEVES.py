def main():
    # import numpy as np
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    from itertools import chain, combinations
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix


    # f = set([0,1,2])
    f = set([0, 1, 2, 3,4,5,6,7,8,9,10,11,12])
    s = []
    alldata = open('./processed.data')
    processed = []
    training = []
    testing = []
    startpoints = [-1, -1, -1, -1, -1]
    training_labels = []
    testing_labels = []
    output = open("output.data",'w')
    matrixfile = open("matrix.data","w")
    

    WEIGHT_MATRIX = [[0, 1.1, 1.2, 1.3, 1.4], [1, 0, 1.1, 1.2, 1.3], [1.0, 1.0, 0, 1.1, 1.2], [1.0, 1.0, 1.0, 0, 1.1], [1.0, 1.0, 1.0, 1.0, 0]]
    PROCEDURE_COSTS = [1.00, 1.00, 1.00, 1.00, 7.27, 5.2, 15.5, 102.9, 87.3, 87.3, 87.3, 100.9, 102.9]

    for z in chain.from_iterable(combinations(f, r) for r in range(len(f)+1)):
        if len(z)>0:
            s.append(z)

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

    # print countDistribution(processed)
    # print getSplit(countDistribution(processed))

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

    # We now retrieve the labels from the last column in the lists and construct our label lists
    for vals in training:
        label = vals[-1]
        if label <= 1:
            training_labels.append(0)
        else:
            training_labels.append(1)

    for vals in testing:
        label = vals[-1]
        if label <= 1:
            testing_labels.append(0)
        else:
            testing_labels.append(1)

    # We use sklearn's StandardScaler to scale our data
    scaler = StandardScaler() 
    scaler.fit(training)
    training = scaler.transform(training)
    testing = scaler.transform(testing)



    print training[0]
    print len(training[0])


    svmClassifier = svm.SVC(gamma=0.01)

    print countDistribution(processed)
    iter_count = 0
    highest = [0,0]
    for permutation in s:
        cost = 0
        temp_training = [[] for i in range(len(training))]
        temp_testing = [[] for i in range(len(testing))]
        for feat in permutation:
            cost += PROCEDURE_COSTS[feat]
            for i in range(len(training)):
                temp_training[i].append(training[i][feat])
            for i in range(len(testing)):
                temp_testing[i].append(testing[i][feat])


        
        iter_count += 1
        svmClassifier.fit(temp_training,training_labels)
        results = svmClassifier.predict(temp_testing)
        score = f1_score(testing_labels, results, average='micro')

        output.write("%d,%s,%s,%d,"%(iter_count, permutation, score, cost))
        cm = confusion_matrix(testing_labels,results)
        output.write(str(cm[0])+ str(cm[1])+"\n")
        # matrixfile.write(str(cm[0])+","+str(cm[1])+"\n")
        if iter_count % 1000 == 0:
            print iter_count, "/", len(s)
        if score > highest[0]:
            highest = [score, permutation, cost]
    print highest
    output.close()
        
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

# to be deprecated
def getConfusionMatrix(results, actual):
    matrix = [[0 for i in range(5)] for i in range(5)]
    totals = [0,0,0,0,0]
    for i in actual:
        totals[int(i)] += 1

    for i in range(len(results)):
        res = int(results[i])
        ac = int(actual[i])
        matrix[res][ac] += 1
        
    

    for i in range(5):
        for j in range(5):
            matrix[i][j] = round(float(matrix[i][j])/float(totals[j]),2)   
    return matrix



# to be deprecated
def getCustomPMScore(matrix, weights):
    # print "henlo"
    # print matrix
    # print weights

    score = 0.0
    for i in range(5):
        for j in range(5):
            matrix[i][j] *= weights[i][j]
            score += matrix[i][j]
    score = score / 20.0
    return 1 - score


if __name__ == "__main__":
    main()  