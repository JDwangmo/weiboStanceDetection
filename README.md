# Weibo Stance Detection
### 微博立场分析


### 实验数据
- 任务A: 
    - evasampledata4-TaskAA.txt:3000条官方数据集,总共6个话题.有标注.但是发现数据中有14条漏标注.分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 600  |600|600|600|600 |
        
    - all_data_2986.csv: 在[evasampledata4-TaskAA.txt]基础上,去除14条漏标数据后的2986条有标注数据,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 600  |600|600|600|586 |
        
    - train_data_2090.csv: 在[all_data_2986.csv]基础上,随机取出2090(占70%)的数据作为训练集,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 441  |416|415|413|405 |
        
    - test_data_896.csv: 在[all_data_2986.csv]基础上,随机取出896(占30%)的数据作为测试集,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:

        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 187  |185|184|181|159 |


### 项目架构
- data_processing:数据预处理包
    - data_util.py: 数据预处理类:包括去除空值,分词,切分训练集和测试集等.
    - preprocessing.py: 已过时.
    
- cnn: 卷积伸进网络分类器
    - randEmbedding_cnn: CNN-rand模型,具体见：https://github.com/JDwangmo/coprocessor#randomembedding_padding_cnn
    
        - 效果:使用 [[train_data/all_data_2986.csv,2986句](https://github.com/JDwangmo/weiboStanceDetection/tree/master/train_data)]做训练.[[train_data/test_data_896.csv,896句](https://github.com/JDwangmo/weiboStanceDetection/tree/master/train_data)].结果位于[[./result/ISCSLP2016/20160618/](https://github.com/JDwangmo/weiboStanceDetection/tree/master/result/20160630/)]
        - 情形1: 1层CNN;
            - 
            - {'full_mode': True,
            - 'need_segmented': True,
            - 'remove_stopword': True,
            - 'replace_number': True,
            - 'sentence_padding_length': 150,
            - 'train_data_count': 2090,
            - 'verbose': 0}
            - {'conv_filter_type': [[100, 2, 300, 'valid'],
            -          [100, 4, 300, 'valid'],
            -          [100, 8, 300, 'valid']],
            - 'earlyStoping_patience': 50,
            - 'embedding_dropout_rate': 0.5,
            - 'input_dim': 14476,
            - 'input_length': 150,
            - 'kmaxpooling_k': 1,
            - 'nb_epoch': 20,
            - 'num_labels': 3,
            - 'output_dropout_rate': 0.5,
            - 'rand_seed': 1337,
            - 'verbose': 1,
            - 'word_embedding_dim': 300}
     
            |句子长度| 词向量长度  | 迭代次数/earlyStop/实际  |训练集结果 | 测试集结果 |结果文件名|运行时间|
            |---|---|---|---|---|---|---|
            |150| 300 | 20/50/20 | \*(\*%) |532(59.375%)|RandEmbeddingCNN_150len_300dim_20epoch_59.csv|8880s|

###相关论文:
- ######1、[Sentiment Analysis: Detecting Valence, Emotions, and Other Affectual States  from Text.pdf](https://raw.githubusercontent.com/JDwangmo/weiboStanceDetection/master/reference/Sentiment-Analysis:Detecting-Valence,Emotions,and-Other-Affectual-States-from-Text.pdf)
        - Sergio Roa, Fernando Nino. FLAIRS 2003.
        - keywords: RNN