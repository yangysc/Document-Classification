# document-classification
Classify documents using Python based on SVM and TF-IDF.


- Two Python librarys(Pandas and liblinear) are needed. On Windows, you can download the liblinear library from http://www.lfd.uci.edu/~gohlke/pythonlibs/#liblinear

- The structures of the data files are:
-- The .data files are formatted "docIdx wordIdx count". 
-- The .label files are simply a list of label id's. 
-- The .map files map from label id's to label names.
- This demo will gives a **accuracy** near **81.3991% (6109/7505)**.
