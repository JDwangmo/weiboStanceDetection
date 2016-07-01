# Weibo Stance Detection
### 微博立场分析


### 实验数据
- 任务A: 
    - `train_data/evasampledata4-TaskAA.txt`:3000条官方数据集,总共6个话题.有标注.但是发现数据中有14条漏标注.分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 600  |600|600|600|600 |
        
    - `train_data/all_data_2986.csv`: 在[evasampledata4-TaskAA.txt]基础上,去除14条漏标数据后的2986条有标注数据,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 600  |600|600|600|586 |
        
    - `train_data/train_data_2090.csv`: 在[all_data_2986.csv]基础上,随机取出2090(占70%)的数据作为训练集,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 441  |416|415|413|405 |
        
    - `train_data/test_data_896.csv`: 在[all_data_2986.csv]基础上,随机取出896(占30%)的数据作为测试集,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:

        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 187  |185|184|181|159 |

    - `train_data/word_count_17587.csv`: 在[all_data_2986.csv]基础上,统计每个词 在各个类别中的次数,每个词有四个统计项：
        1. FAVOR：	在favor类别中的出现的次数
        2. AGAINST：在AGAINST类别中的出现的次数
        3. NONE	： 在NONE类别中的出现的次数
        4. FREQ	： 在所有类别中的出现的次数，即FAVOR+AGAINST+NONE
        5. SUPPORT： 最高词频词频项/（FREQ）.
        - 处理程序:[data_processing/data_util.py]. 产生17587个词:
        
    - `train_data/word_count_17211.csv`: 在[all_data_2986.csv]基础上,`train_data/word_count_17587.csv`一样，不过增加繁体转简体，大小写，移除url等处理。统计每个词 在各个类别中的次数,每个词有四个统计项：
        1. FAVOR：	在favor类别中的出现的次数
        2. AGAINST：在AGAINST类别中的出现的次数
        3. NONE	： 在NONE类别中的出现的次数
        4. FREQ	： 在所有类别中的出现的次数，即FAVOR+AGAINST+NONE
        5. SUPPORT： 最高词频词频项/（FREQ）.
        - 处理程序:[data_processing/data_util.py]. 产生17211个词:
    
    - `train_data/训练(3000)一半.xlsx`: 该文件在`train_data/evasampledata4-TaskAA.txt`基础上，将原句子进行截断处理，只取原句子的一半句子,3000句。
    - `train_data/测试(896)一半.xlsx`:该文件在`train_data/test_data_896.csv`基础上，将原句子进行截断处理，只取原句子的一半句子,896句。
    
    - `train_data/all_data_half_2986.csv`:该文件在`train_data/all_data_half_2986.csv`基础上，将原句子进行截断处理，只取原句子的一半句子,2986句。
    
    - `train_data/train_data_half_2090.csv`:该文件在`train_data/train_data_2090.csv`基础上，将原句子进行截断处理，只取原句子的一半句子,2090句。
    - `train_data/test_data_half_896.csv`:该文件在`train_data/test_data_half_896.csv`基础上，将原句子进行截断处理，只取原句子的一半句子,896句。

### 项目架构
- data_processing:数据预处理包
    - data_util.py: 数据预处理类:包括去除空值,分词,切分训练集和测试集等.
    - preprocessing.py: 已过时.
    
- cnn: 卷积伸进网络分类器
    - randEmbedding_cnn: CNN-rand模型,具体见：https://github.com/JDwangmo/coprocessor#randomembedding_padding_cnn
    
        - 效果:
        - 情形1: 1层CNN;使用 [[train_data/all_data_2986.csv,2986句](https://github.com/JDwangmo/weiboStanceDetection/tree/master/train_data)]做训练.[[train_data/test_data_896.csv,896句](https://github.com/JDwangmo/weiboStanceDetection/tree/master/train_data)].结果位于[[./result/ISCSLP2016/20160618/](https://github.com/JDwangmo/weiboStanceDetection/tree/master/result/20160630/)]句子最长长度为：175,句子最短长度为：3,句子平均长度为：44.
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
            
        - 情形2: 1层CNN;使用 [[train_data/train_data_half_2090.csv,2090句](https://github.com/JDwangmo/weiboStanceDetection/tree/master/train_data)]做训练.[[train_data/test_data_half_896.csv,896句](https://github.com/JDwangmo/weiboStanceDetection/tree/master/train_data)],使用截断后的句子，句子最长长度为：93，句子最短长度为：2，句子平均长度为：27。
            - {'add_unkown_word': True,
             'full_mode': True,
             'mask_zero': True,
             'need_segmented': True,
             'padding_mode': 'center',
             'remove_stopword': True,
             'replace_number': True,
             'sentence_padding_length': 93,
             'train_data_count': 2090,
             'train_data_dict_size': 10543,
             'verbose': 0}
            {'conv_filter_type': [[100, 2, 300, 'valid'],
                                  [100, 4, 300, 'valid'],
                                  [100, 8, 300, 'valid']],
             'earlyStoping_patience': 50,
             'embedding_dropout_rate': 0.5,
             'input_dim': 10544,
             'input_length': 93,
             'kmaxpooling_k': 1,
             'nb_epoch': 100,
             'num_labels': 3,
             'output_dropout_rate': 0.5,
             'rand_seed': 1337,
             'verbose': 1,
             'word_embedding_dim': 50}
            - 
            |句子长度| 词向量长度  | 迭代次数/earlyStop/实际  |训练集结果 | 测试集结果 |结果文件名|运行时间|
            |---|---|---|---|---|---|---|
            |93| 50 | 100/50/100 | \*(\*%) |479(53.4598%)|\*.csv|8880s|
            
- cue_pharse :关键词匹配
     - cue_pharse.py:
        - 只取support>=0.6 & frequency>=5  的取出来做候选，共1272个
        1. support>=0.8 & frequency>=5 （共369个）和 support>=0.75 & frequency>=10 的直接出线 （增加共40个）,合起来共406个。
        2. 剩余的 866个，和全部非自身（也即是1281个）的候选，两两组合，如果有达到上述（1）标注的也出线，这个统计看看有多少。
        3. 出线后的词，直接用于第一轮（基于规则的判断），多个词同时出现在同一句会很正常，可以就取最高support的。（对于（2）的词，只在他大于入选的组合词也在时才纳入比较）
        - （1）全句，训练、测试
        - （2）半句，训练、测试
        - （3）全句训练，半句测试 （比（1）和（2）增加了训练语料的保留，当然会顺带有些副作用，比如那些有转折的也一起参加训练，好坏难说，交给程序结果来看吧）

    

###相关论文:
- ######1、[Sentiment Analysis: Detecting Valence, Emotions, and Other Affectual States  from Text.pdf](https://raw.githubusercontent.com/JDwangmo/weiboStanceDetection/master/reference/Sentiment-Analysis:Detecting-Valence,Emotions,and-Other-Affectual-States-from-Text.pdf)
        - Sergio Roa, Fernando Nino. FLAIRS 2003.
        - keywords: RNN