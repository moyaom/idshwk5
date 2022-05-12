from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math


class Domain:
    def __init__(self, _name, _label, _length, _nums, _entropy, _seg):
        self.name = _name
        self.label = _label
        self.length = _length
        self.nums = _nums
        self.entropy = _entropy
        self.seg = _seg

    def returnData(self):
        return [self.length, self.nums, self.entropy, self.seg]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0


def cal_nums(s):
    nums = 0
    for i in s:
        if i.isdigit():
            nums += 1
    return nums


def cal_entropy(s):
    h = 0.0
    sumLetter = 0
    letter = [0] * 26
    s = s.lower()
    for i in range(len(s)):
        if s[i].isalpha():
            letter[ord(s[i]) - ord('a')] += 1
            sumLetter += 1
    for i in range(26):
        p = 1.0 * letter[i] / sumLetter
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


def cal_seg(s):
    nums = 0
    for i in s:
        if i == '.':
            nums += 1
    return nums


def initData(filename, domainlist):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            tokens = line.split(',')
            name = tokens[0]
            if len(tokens) > 1:
                label = tokens[1]
            else:
                label = '?'
            length = len(name)
            num = cal_nums(name)
            entropy = cal_entropy(name)
            seg = cal_seg(name)
            domainlist.append(Domain(name, label, length, num, entropy, seg))


def main():
    domainlist1 = []
    initData('train.txt', domainlist1)
    featureMatrix = []
    labelList = []
    for i in domainlist1:
        featureMatrix.append(i.returnData())
        labelList.append(i.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    domainlist2 = []
    initData('test.txt', domainlist2)
    with open('result.txt', 'w') as f:
        for i in domainlist2:
            if clf.predict([i.returnData()])[0] == 0:
                f.write('%s,notdga' % i.name)
            else:
                f.write('%s,dga' % i.name)
            f.write('\n')


if __name__ == '__main__':
    main()
