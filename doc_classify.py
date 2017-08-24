import pandas as pd
import math
from liblinearutil import *
import time

# 读取数据
def loadOriginData(src='train'):
    # train.data
    dataSrc = r'%s.data' % src
    # train.label
    labelSrc = r'%s.label' % src
    label = pd.read_table(labelSrc, sep=' ', names=['label'])
    # train.map
    mapSrc = r'%s.map' % src

    # 每个文档拥有的terms
    doc2term = {}
    # 每个term出现在哪些文档
    term2doc = {}
    # 每个类别下有哪些docs
    cate2docs = {}
    # TF值
    TF = {}
    with open(dataSrc, 'r') as f:
        for line in f:
            str_docIdx, str_wordIdx, str_cnt = line.split()
            docIdx = int(str_docIdx)
            wordIdx = int(str_wordIdx)
            cnt = int(str_cnt)
            # update 数据结构
            doc2term.setdefault(docIdx, []).append(wordIdx)
            term2doc.setdefault(wordIdx, []).append(docIdx)
            TF.setdefault(docIdx, {})[wordIdx] = cnt
    # 统计每个类别下有哪些文档
    with open(labelSrc, 'r') as f:
        for line_index, line in enumerate(f, 1):
            labelVal = int(line.strip())
            cate2docs.setdefault(labelVal, []).append(line_index)
    return TF, doc2term, term2doc, cate2docs, label


#  特征选择
def featureSel(doc2term, term2doc, cate2docs):
    # CHI衡量的是特征项ti和类别Cj之间的关联程度, A,B, C, D是四个统计量
    CHI_cat2term = {}
    # N：total  number of documents
    N = len(doc2term)
    # A + B + C + D = N
    # A： term出现在某类别中的文档总数
    A = {}
    # B: term出现在除某类别外的其他文档数
    B = {}
    # C:  该类别中不包含term的文档总数
    C = {}
    # D: 其他类别中不包含term的文档总数
    D = {}
    DF = {}
    # 所有类别
    categories = list(cate2docs.keys())
    # 停用词词汇表
    stopwords = {}
    stopwordsSrc = r'stopwords.txt'
    with open(stopwordsSrc) as f:
        for line in f:
            stopwords[line.strip()] = True
    # 训练数据数据词汇表
    vocSrc = r'vocabulary.txt'
    voc = pd.read_table(vocSrc, names=['voc'])
    # 保存所有的特征
    features = set()
    # 计算一个类别标签下各个词的CHI
    for category in categories:
        # 属于第category类的文档为docs
        docs = cate2docs[category]
        sumVal = 0
        for term in term2doc:
            # 如果是停用词, 则将CHI置零
            if stopwords.get(voc['voc'][term - 1], False):
                CHI_cat2term.setdefault(category, {})[term] = 0
                continue
            # 属于某类且包含term
            AVal = len(set(term2doc[term]).intersection(set(docs)))
            # 不属于某类但包含term
            BVal = len(term2doc[term]) - AVal
            # 属于某类，但不包含term
            CVal = len(docs) - AVal
            # 不属于某类， 不包含term
            DVal = N - AVal - BVal - CVal
            CHIVal = N * (AVal * DVal - CVal * BVal)**2 / ((AVal + CVal) * (BVal + DVal) * (AVal + BVal) * (CVal + DVal))
            # CHIVal = math.log(AVal * N / ((AVal + CVal) * (AVal + BVal)))
            A.setdefault((term, category), AVal)
            B.setdefault((term, category), BVal)
            C.setdefault((term, category), CVal)
            D.setdefault((term, category), DVal)

            CHI_cat2term.setdefault(category, {})[term] = CHIVal
            DF[term] = AVal + BVal
            sumVal += CHIVal
        # 选出类别中CHI高于平均值的词
        terms = CHI_cat2term[category]
        meanVal = sumVal / len(terms)
        for term in terms:
            if CHI_cat2term[category][term] > meanVal:
                features.add(term)
    # for feature in features:
    #     print(voc['voc'][feature])
    print('There are %d features in VSM model.\n' % len(features))
    return features,  DF


def buildSVMData(TF, DF, features, N, label, cate2docs, doc2terms):
    isFeatures = dict(zip(features, [True] * len(features)))
    categories = list(cate2docs.keys())
    # 如果是训练样本， 则计算归一化缩放因子，并返回
    # y： label值
    y = [0] * N
    # x: 稀疏矩阵
    x = []
    for i in range(N):
        x.append({})
    for category in categories:
        for doc in cate2docs[category]:
            # 给y进行标记类别
            y[doc - 1] = label.iat[doc - 1, 0]
            scale_factor = -100
            for term in doc2terms[doc]:
                if isFeatures.get(term, False):  # 如果term是特征
                    # TF值
                    TFVal = TF[doc].get(term, 0)
                    # TF-IDF值
                    tf_idf = TFVal * math.log(N / DF[term])
                    x[doc - 1][term] = tf_idf
                    # 更新特征最大值
                    if scale_factor < tf_idf:
                        scale_factor = tf_idf
            alpha = 0
            # 按一篇文档中特征词最大的tf-idf, 对该文档中的所有特征词进行归一化
            for term in doc2terms[doc]:
                if isFeatures.get(term, False):  # 如果term是特征
                    # x[doc - 1][term] = alpha + (1 - alpha) * x[doc - 1][term] / scale_factor
                    x[doc - 1][term] /= scale_factor
    print("Data for SVM has been built.\n")
    return x, y

# 计算DF
def getDF(doc2term, term2doc, cate2docs):
    DF = {}
    for term in term2doc:
        DF[term] = len(term2doc[term])
    return DF

if __name__ == '__main__':
    start = time.time()
    # # 主程序
    TF, doc2term, term2doc, cate2docs, label = loadOriginData()
    # 特征选择
    features, DF = featureSel(doc2term, term2doc, cate2docs)
    # 读取数据(train.data)
    TF, doc2term, term2doc, cate2docs, label = loadOriginData()
    # 特征选择
    features, DF = featureSel(doc2term, term2doc, cate2docs)
    # build SVM model
    x, y = buildSVMData(TF, DF, features, len(doc2term), label, cate2docs, doc2term)
    # 读取测试数据(test.data)
    TF_test, doc2term_test, term2doc_test, cate2docs_test, label_test = loadOriginData(src='test')
    DF_test = getDF(doc2term_test, term2doc_test, cate2docs_test)
    # TF, DF, features, len(doc2term), label, cate2docs, doc2term, scales)
    x_test, y_test = buildSVMData(TF_test, DF_test, features, len(doc2term_test), label_test, cate2docs_test, doc2term_test)

    print("处理数据使用了 %s s时间。\n" % (time.time() - start))
    # # 调用 liblinear 库进行分类
    prob = problem(y, x)
    param = parameter('-s 0 -c 4 -B 1')
    # 训练
    m = train(prob, param)
    # 预测test.data
    p_label, p_acc, p_vals = predict(y_test, x_test, m, '-b 1')
    # 评价
    ACC, MSE, SCC = evaluations(y_test, p_label)
    print('ACC:\n', ACC)
    print('MSE', MSE)
    print('SCC', SCC)
    # 统计每类中错误率
    categoriesErrs = {}
    for doc_index, doc_label in enumerate(y_test):
        if doc_label != int(p_label[doc_index]):
            cateogory = label_test.iat[doc_index, 0]
            categoriesErrs.setdefault(cateogory, []).append(doc_index + 1)
    # with open('outcome.txt', 'wb') as f:
    print("错误分类的样本为：\n")
    for categoryErr in categoriesErrs:
        numOfErr = len(categoriesErrs[categoryErr])
        print('第%d类共 %d样本, 被错分的个数为 %d, 比例为 %f %%.\n' % (categoryErr,len(cate2docs_test[categoryErr]), numOfErr, numOfErr/len(cate2docs_test[categoryErr])))

    end = time.time()
    print("Total time cost is  %s s.\n" % (end - start))



