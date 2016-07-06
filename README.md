# Weibo Stance Detection
### [微博立场分析](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html)


### 实验数据
- 任务A: 
    - `train_data/evasampledata4-TaskAA.txt`:3000条官方数据集,总共6个话题.有标注.但是发现数据中有14条漏标注.分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 600  |600|600|600|600 |
        
    - `train_data/TaskAA_all_data_3000.csv`: 在[evasampledata4-TaskAA.txt，3000句]基础上,去除TEXT字段的空值数据（TEXT字段无空值，不需去除，剩下3000句）,已提供分词字段[WORD].处理程序:[data_processing/data_util.py]，有3000句。
    
    - `train_data/TaskAA_all_data_2986.csv`: 在[evasampledata4-TaskAA.txt]基础上,去除14条漏标数据后的2986条有标注数据,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 600  |600|600|600|586 |
        
    - `train_data/TaskAA_train_data_2090.csv`: 在[all_data_2986.csv]基础上,随机取出2090(占70%)的数据作为训练集,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:
    
        |话题(TARGET) |春节放鞭炮  |开放二胎|俄罗斯在叙利亚的反恐行动|深圳禁摩限电|IphoneSE|
        |---|---|---|---|---|---|
        |个数 | 441  |416|415|413|405 |
        
    - `train_data/TaskAA_test_data_896.csv`: 在[all_data_2986.csv]基础上,随机取出896(占30%)的数据作为测试集,已提供分词字段[WORD].处理程序:[data_processing/data_util.py].分布如下:

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
    
    - `train_data/test_data_half_896.csv`:该文件在`train_data/test_data_896.csv`基础上，将原句子进行截断处理，只取原句子的一半句子,896句。
    
    - `train_data/test_data_Mhalf_896.csv`:该文件在`train_data/test_data_896.csv`基础上，将原句子进行截断处理，只取原句子的一半句子（句子长度<=4的不切，>4的切一半）,896句,已经分词。
    
    - `train_data/evasampledata4-TaskAR.txt`:官方无标注训练数据，用于任务A，3000句，TEXT字段有空值数据。
    - `train_data/TaskAR_all_data_2997.csv`:该文件在`train_data/evasampledata4-TaskAR.txt，3000句`基础上，进行了去除空值（原来3000句，由于TEXT字段有三个空值，去除掉，剩下2997句）和分词，2997句。
    
    - `train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt`:官方测试数据集，未分词，因为有部分句子已标注，所以自行添加了一个PREDICT字段，总共15000句，TEXT字段无空值。
    
    - `train_data/TaskA_all_testdata_15000.csv`:官方测试数据集，在`train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt`基础上，,已分好词，含15000句,分完词之后出现34句是空值的，剩下有14966句。
    - `train_data/TaskA_all_testdata_14966.csv`:官方测试数据集，在`train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt`基础上，将句子分词，已去除WORDS空串，剩下14966句。
    
    - `train_data/TaskA_testdata_half.csv`:官方测试数据集，在`train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt`基础上，将句子切分成一半（句子长度<=4的不切，>4的切一半），切半后句子未分词。
    
    - `train_data/TaskA_all_testdata_half_15000.csv`:官方测试数据集，在`train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt`基础上，将句子分词,有15000句。
    - `train_data/TaskA_all_testdata_half_14966.csv`:官方测试数据集，在`train_data/TaskA_testdata_half.csv`基础上，将句子分词,有，不过其中有34句，在分词之后为空串。

### 项目架构
- data_processing:数据预处理包
    - data_util.py: 数据预处理类:包括去除空值,分词,切分训练集和测试集等.
    - preprocessing.py: 已过时.
    
- cnn: 卷积伸进网络分类器
    - randEmbedding_cnn: CNN-rand模型,具体见：https://github.com/JDwangmo/coprocessor#randomembedding_padding_cnn
        - 实验设置1：
            - 数据：   
                - 训练全： [train_data/train_data_full_2090.csv]. 句子最长长度为：175,句子最短长度为：3,句子平均长度为：44,字典数量：14110。
                - 训练半H：[train_data/train_data_half_2090.csv]. 所有句子都切成一半，结果：句子最长长度为：93，句子最短长度为：2，句子平均长度为：27,字典数量：10543。
                - 训练半M：[train_data/TaskAA_train_data_Mhalf_2090.csv]. 切分方案为：>4切一半，<=4不切全保留，结果：句子最长长度为：93，句子最短长度为：3，句子平均长度为：30,字典数量：11241。
                - 训练final：[train_data/TaskAA_train_data_final_3000.csv]. 3000句训练数据。句子最长长度为：175,句子最短长度为：3,句子平均长度为：45,字典数量：17308.
                - 测试全：[train_data/test_data_full_896.csv].
                - 测试半H：[train_data/test_data_half_896.csv].
                - 测试半M：[train_data/TaskAA_test_data_Mhalf_896.csv].
                
            - 通用设置：
            - {'add_unkown_word': True,
            - 'full_mode': True,
            - 'mask_zero': True,
            - 'need_segmented': True,
            - 'padding_mode': 'center',
            - 'remove_stopword': True,
            - 'replace_number': True,
            - 'train_data_count': 2090,
            - 'verbose': 0}
            - 'embedding_dropout_rate': 0.5,
            - 'embedding_init use rand': False,
            - 'kmaxpooling_k': 1,
            - 'num_labels': 3,
            - 'output_dropout_rate': 0.5,
            - 'rand_seed': 1337,
            - 'verbose': 1,
            - 具体如下：
            
            | 训练数据  |优化算法|句子补齐长度|卷积核类型|rand_seed|词向量维度|迭代次数(最大/early stop/最终)|测试数据|准确率|F1值宏平均(Favor,Against,None)|结束时训练误差|模型名|
            |---|---|---|---|---|---|---|---|---|---|---|---|
            | 训练半M  | 93  | adadelta |248  |0 | 50  | 20/50/20  | 测试半M  | 345(0.385045)  | 0.278002([ 0.55600322，0.， 0.]) |1.0460|RandEmbeddingCNN_adadelta_93len_50dim_1k_Mhalftrain_20epoch_0seed.pkl|
            | 训练半M  | 93  | adadelta|248  |100 | 50  | 20/50/20  | 测试半M  | 345(0.385045)  | 0.278002([ 0.55600322  0.          0.        ]) |  1.0986|RandEmbeddingCNN_adadelta_93len_50dim_1k_Mhalftrain_20epoch_100seed.pkl|
            | 训练半M  | 93  | adadelta|248  |400 | 50  | 20/50/20  | 测试半M  | 477(0.532366)  | 0.591280([ 0.56573705  0.61682243  0.        ]) |  0.9505|RandEmbeddingCNN_adadelta_93len_50dim_1k_Mhalftrain_20epoch_400seed.pkl|
            | 训练半M  | 93  |adadelta| 248  |600 | 50  | 20/50/20  | 测试半M  | 458(0.511161)  | 0.569670([ 0.56374269  0.57559682  0.        ]) |  1.0427 |RandEmbeddingCNN_adadelta_93len_50dim_1k_Mhalftrain_20epoch_600seed.pkl|
            | 训练半M  | 93  | adadelta|248  |800 | 50  | 20/50/20  | 测试半M  | 484(0.540179)  | 0.599057([ 0.57025921  0.62785388  0.        ]) |  0.9711 |RandEmbeddingCNN_adadelta_93len_50dim_1k_Mhalftrain_20epoch_800seed.pkl|
            | 训练半M  | 93  | adadelta|248  |1000 | 50  | 20/50/20  | 测试半M  | 482(0.537946)  | 0.593691([ 0.55221745  0.63516484  0.        ]) |  0.9735 |RandEmbeddingCNN_adadelta_93len_50dim_1k_Mhalftrain_20epoch_1000seed.pkl|
            | 训练全  | 93  | adadelta|248  |1000 | 50  | 30/50/30  | 测试半M  | 484(0.540179)  | 0.601959([ 0.59649123，0.60742706，0.]) |  0.9252 |RandEmbeddingCNN_adadelta_93len_50dim_1k_fulltrain_30epoch_1000seed.pkl|
            | 训练全  | 93  | adadelta|248  |1000 | 300  | 30/50/30  | 测试半M  | 526(0.587054)  | 0.641443([ 0.62879789,0.65408805,0.23333333]) |  0.\* |RandEmbeddingCNN_adadelta_93len_300dim_1k_fulltrain_30epoch_1000seed.pkl|
            | 训练全  | 93  | adadelta|248  |2000 | 300  | 30/50/30  | 测试半M  | 524(0.584821)  | 0.631070([ 0.60892388,0.65321564,0.27848101]) |  0.6883 |RandEmbeddingCNN_adadelta_93len_300dim_1k_fulltrain_30epoch_2000seed.pkl|
            | 训练全  | 93  | adadelta|248  |4000 | 300  | 30/50/30  | 测试半M  | 528(0.589286)  | 0.642195([ 0.61849711,0.66589327,0.22689076]) |  0.7685 |RandEmbeddingCNN_adadelta_93len_300dim_1k_fulltrain_30epoch_4000seed.pkl|
            | 训练全  | 93  | adadelta|248  |5000 | 300  | 30/50/30  | 测试半M  | 527(0.588170)  | 0.628012([ 0.58860759,0.66741573,0.32592593]) |  0.6551 |RandEmbeddingCNN_adadelta_93len_300dim_1k_fulltrain_30epoch_5000seed.pkl|
            | 训练全  | 93  | 248  |1000 | 300  | 30/50/30  | 测试半M  | 361(0.402902)  | 0.448278([ 0.43196005,0.46459627,0.01075269]) |  0.9302 |RandEmbeddingCNN_sgd_93len_300dim_1k_fulltrain_30epoch_1000seed.pkl|
            | 训练final  | 93  | 248  |1000 | 300  | 40/50/40  | 测试全  | 707(0.789062)  | 0.885205([ 0.797219,0.97319035,0.0]) |  0.6940 |RandEmbeddingCNN_adadelta_93len_300dim_1k_finaltrain_40epoch_1000seed.pkl|
            | 训练final  | 93  | 248  |1000 | 300  | 60/50/60  | 测试全  | 709(0.791295)  | 0.886818([ 0.79907085,0.97456493,0.01086957]) |  0.6909 |RandEmbeddingCNN_adadelta_93len_300dim_1k_finaltrain_60epoch_1000seed.pkl|

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
            
    - pretrainEmbedding_cnn: CNN-nonstatic模型,具体见：https://github.com/JDwangmo/coprocessor#randomembedding_padding_cnn
        - 实验设置1：
            - 数据：   
                - 训练全： [train_data/train_data_full_2090.csv]. 句子最长长度为：175,句子最短长度为：3,句子平均长度为：44,字典数量：14110。
                - 训练半：[train_data/train_data_half_2090.csv]. 句子最长长度为：93，句子最短长度为：2，句子平均长度为：27,字典数量：10543。
                - 测试全：[train_data/test_data_full_896.csv].
                - 测试半：[train_data/test_data_half_896.csv].
            - 通用设置：
            - {'add_unkown_word': True,
            - 'full_mode': True,
            - 'mask_zero': True,
            - 'need_segmented': True,
            - 'padding_mode': 'center',
            - 'remove_stopword': True,
            - 'replace_number': True,
            - 'train_data_count': 2090,
            - 'verbose': 0}
            - 'embedding_dropout_rate': 0.5,
            - 'embedding_init use rand': False,
            - 'kmaxpooling_k': 1,
            - 'num_labels': 3,
            - 'output_dropout_rate': 0.5,
            - 'rand_seed': 1337,
            - 'verbose': 1,
        
            | 训练数据  |pretrain词向量|rand_seed|句子补齐长度|卷积核类型|词向量维度|迭代次数(最大/early stop/最终)|测试数据|准确率|F1值|结束时训练误差|
            |---|---|---|---|---|---|---|---|---|---|---|
            | 训练全  |train_data_full_20963_50dim_50iter_cbow.gem  |1337| 93  | 248  | 50  | 200/100/186  | 测试全  | 338(0.377232)  | 0.278368([ 0.54615385  0.01058201  0.27807487])  |0.9779| 
            | 训练半  | train_data_full_20963_50dim_50iter_cbow.gem  |1337| 93  | 248  | 50  | 200/100/200  | 测试全  | 467(0.5212052)  | 0.568713([ 0.59142212  0.54600302  0.19753086])  | 0.8578|
            | 训练全  | train_data_full_20963_50dim_50iter_cbow.gem  |1337| 93  | 248  | 300  | 200/100/200  | 测试全  | \*(0.5212052)  | \*([ 0.59142212  0.54600302  0.19753086])  | \*|
            | 训练半  | train_data_full_20963_50dim_50iter_cbow.gem  |1337| 93  | 248  | 300  | 200/100/200  | 测试全  | \*(0.5212052)  | \*([ 0.59142212  0.54600302  0.19753086])  | \*|
            | 训练全  | train_data_full_20963_50dim_50iter_cbow.gem  |1337| 150  | 248  | 300  | 200/100/200  | 测试全  | \*(0.5212052)  | \*([ 0.59142212  0.54600302  0.19753086])  | \*|

- cue_pharse :关键词匹配

     - cue_pharse.py:
     1. 使用[train_data/TaskAA_train_data_full_2090.csv]进行统计,总共有14110个词[train_data/word_count_14110.csv]。
        - 方案3(L版本)：条件最宽松
            - 只取support>=0.55 & frequency>=5  的取出来做候选，共1171个.[train_data/candiate_keywords_1171.csv]
            1. support>=0.75 & frequency>=5 （共\*个）和 support>=0.7 & frequency>=10 的直接出线 （增加共\*个）,合起来共368个。[train_data/selected_keywords_L_368.csv]
            2. 全部非自身（也即是1171个）的候选，两两组合(1370070个)，如果有达到上述（1）标注的也出线，这个统计看看有多少(1522个)。[train_data/selected_2gram_Lc_1522.csv]
            3. 出线后的词，直接用于第一轮（基于规则的判断），多个词同时出现在同一句会很正常，可以就取最高support的。（对于（2）的词，只在他大于入选的组合词也在时才纳入比较）
     2. 使用[train_data/TaskAA_all_data_2986.csv]进行统计,总共有17211个词[train_data/word_count_17211.csv]。
        - 方案1(H版本)：条件最严格
            - 只取support>=0.6 & frequency>=5  的取出来做候选，共1272个[train_data/candiate_keywords_1272.csv].
            1. support>=0.85 & frequency>=5 （共\*个）和 support>=0.8 & frequency>=10 的直接出线 （增加共\*个）,合起来共188个。[train_data/selected_keywords_H_188.csv]
                - 2A. 剩余的 1084个，和全部非自身（也即是1272个）的候选，两两组合(1377764个)，如果有达到上述（1）标注的也出线，这个统计看看有多少(565个)。[train_data/selected_2gram_H_565.csv]
                - 2B. 全部非自身（也即是1272个）的候选，两两组合(1616712个)，如果有达到上述（1）标注的也出线，这个统计看看有多少(866个)。[train_data/selected_2gram_Hc_866.csv]
            3. 出线后的词，直接用于第一轮（基于规则的判断），多个词同时出现在同一句会很正常，可以就取最高support的。（对于（2）的词，只在他大于入选的组合词也在时才纳入比较）
            - （1）全句，训练、测试
            - （2）半句，训练、测试
            - （3）全句训练，半句测试 （比（1）和（2）增加了训练语料的保留，当然会顺带有些副作用，比如那些有转折的也一起参加训练，好坏难说，交给程序结果来看吧）    
            
        - 方案2(M版本)：条件中等
            - 只取support>=0.6 & frequency>=5  的取出来做候选，共1272个[train_data/candiate_keywords_1272.csv].
            1. support>=0.8 & frequency>=5 （共369个）和 support>=0.75 & frequency>=10 的直接出线 （增加共40个）,合起来共406个。[train_data/selected_keywords_M_406.csv]
                - 2A. 剩余的 866个，和全部非自身（也即是1272个）的候选，两两组合(726141个)，如果有达到上述（1）标注的也出线，这个统计看看有多少(479个)。[train_data/selected_2gram_M_479.csv]
                - 2B. 全部非自身（也即是1272个）的候选，两两组合(808356个)，如果有达到上述（1）标注的也出线，这个统计看看有多少(644个)。[train_data/selected_2gram_Mc_644.csv]
            3. 出线后的词，直接用于第一轮（基于规则的判断），多个词同时出现在同一句会很正常，可以就取最高support的。（对于（2）的词，只在他大于入选的组合词也在时才纳入比较）
            - （1）全句，训练、测试
            - （2）半句，训练、测试
            - （3）全句训练，半句测试 （比（1）和（2）增加了训练语料的保留，当然会顺带有些副作用，比如那些有转折的也一起参加训练，好坏难说，交给程序结果来看吧）
                    
        - 方案3(L版本)：条件最宽松
            - 只取support>=0.55 & frequency>=5  的取出来做候选，共1593个.[]
            1. support>=0.75 & frequency>=5 （共\*个）和 support>=0.7 & frequency>=10 的直接出线 （增加共\*个）,合起来共541个。[train_data/selected_keywords_L_541.csv]
            2. 全部非自身（也即是1593个）的候选，两两组合(2536056个)，如果有达到上述（1）标注的也出线，这个统计看看有多少(2780个)。[train_data/selected_2gram_Lc_2780.csv]
            3. 出线后的词，直接用于第一轮（基于规则的判断），多个词同时出现在同一句会很正常，可以就取最高support的。（对于（2）的词，只在他大于入选的组合词也在时才纳入比较）
            - （1）全句，训练、测试
            - （2）半句，训练、测试
            - （3）全句训练，半句测试 （比（1）和（2）增加了训练语料的保留，当然会顺带有些副作用，比如那些有转折的也一起参加训练，好坏难说，交给程序结果来看吧）
    
    

###相关论文:
- ######1、[Sentiment Analysis: Detecting Valence, Emotions, and Other Affectual States  from Text.pdf](https://raw.githubusercontent.com/JDwangmo/weiboStanceDetection/master/reference/Sentiment-Analysis:Detecting-Valence,Emotions,and-Other-Affectual-States-from-Text.pdf)
        - Sergio Roa, Fernando Nino. FLAIRS 2003.
        - keywords: RNN