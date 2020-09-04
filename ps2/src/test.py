
svm = open("output/p06_SVM_predictions", "r").readlines()
nb = open("output/p06_naive_bayes_predictions", "r").readlines()
text = open("ds6_test.tsv", "r").readlines()
svm = [line.strip() for line in svm]
nb = [line.strip() for line in nb]
text = [line.strip() for line in text]
for i in range(len(svm)):
    if svm[i] != nb[i]:
        print(i, text[i])