
def accuracy(truelabels, predictlabels):
    correct = 0
    for true, pred in zip(truelabels, predictlabels):
        if true == pred:
            correct += 1
    accuracy = correct / len(truelabels)
    return accuracy

def macro(truelabels, predictlabels):
    classes = set(truelabels)
    f1scores = []
    for label in classes:
        tp = 0
        fp = 0
        fn = 0
        for true, pred in zip(truelabels, predictlabels):
            if true == label and pred == label:
                tp += 1
            elif true != label and pred == label:
                fp += 1
            elif true == label and pred != label:
                fn += 1
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1scores.append(f1)
    macrof1 = sum(f1scores) / len(f1scores)
    return macrof1