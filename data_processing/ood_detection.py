#encoding=utf8
from __future__ import print_function


__author__ = 'jdwang'
__date__ = 'create date: 2016-07-05'
__email__ = '383287471@qq.com'


from data_processing.data_util import DataUtil
import io
final_test_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/result_L.txt'
output_file_path = 'result/TaskA_all_testdata_15000_ood.csv'

data_util = DataUtil()

final_test_data = data_util.load_data(final_test_file_path)
print(final_test_data.head())
print(final_test_data.shape)
# 春节放鞭炮
targetA_keywords=[u'鞭炮',u'烟花',u'炮',u'爆竹']
# 开放二胎
targetB_keywords=[u'二胎',u'二孩',u'生',u'孩子']
counter1 =0
counter2 =0
with io.open(output_file_path,'w',encoding='utf8') as fout:
    fout.write(u'ID\tTARGET\tTEXT\tSTANCE\n' )
    for id,target,text,stance in final_test_data.values:
        # print(id)
        # print(target)
        # print(text)
        # print(stance)
        if target == u'春节放鞭炮':
            # print(','.join(data_util.segment_sentence(text).split()))
            # is_in_target = [item in targetA_keywords for item in data_util.segment_sentence(text).split()]
            is_in_target = [text.__contains__(item) for item in targetA_keywords]
            # print(is_in_target)
            # quit()
            # print(sum(is_in_target))
            if sum(is_in_target)>0:
                fout.write(u'%s\t%s\t%s\t%s\n'%(id,target,text,stance))
                # print(id)
            else:
                fout.write(u'%s\t%s\t%s\t%s\n'%(id,target,text,u'NONE'))
                counter1+=1
                # print(id)
        elif target == u'开放二胎':
            # print(','.join(data_util.segment_sentence(text).split()))
            # is_in_target = [item in targetB_keywords for item in data_util.segment_sentence(text).split()]
            is_in_target = [text.__contains__(item) for item in targetB_keywords]

            # print(is_in_target)
            # print(sum(is_in_target))
            if sum(is_in_target)>0:
                fout.write(u'%s\t%s\t%s\t%s\n'%(id,target,text,stance))
                # print(id)
            else:
                fout.write(u'%s\t%s\t%s\t%s\n'%(id,target,text,u'NONE'))
                # fout.write(u'%s\t%s\t%s\t%s\n'%(id,target,text,u'NONE'))
                counter2+=1
                # print(id)
        else:
            fout.write(u'%s\t%s\t%s\t%s\n' % (id, target, text, stance))

print(counter1)
print(counter2)
quit()