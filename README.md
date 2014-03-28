oneseg
======

一枚精巧的中文分词工具，作为我博士后工作成果之一。包含了我博后期间三篇论文的思想。

使用部分标注语料训练模型
-------------------------------

本人文章 **Improving Chinese Word Segmentation Using Partially Annotated Sentences** 有所实践。 此外可以参考《Discriminative Learning with Natural Annotations: Word Segmentation as a Case Study》，本工具基于以上两篇文章的思路实现。

感知器的额外正则化
----------------------------------

本人文章 **Regularized Structured Perceptron for Chinese Word Segmetnation, POS Tagging and Parsing** 本工具实现了再平均的正则化。 

基于分布式表示作为特征
-----------------------------

本人文章 《 **基于自动编码器的中文词汇特征无监督学习** 》， 原文中用词向量设计特征。 本工具使用字的bigram的向量来设计特征。

TODO
-------

* 加入单元测试
* 支持动态增加标注集
* 加入对词性标注的支持
* weight decay的正则化， dropout的正则化
