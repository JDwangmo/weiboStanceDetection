#encoding=utf8
from __future__ import print_function


__author__ = 'jdwang'
__date__ = 'create date: 2016-07-05'
__email__ = '383287471@qq.com'


from data_processing.data_util import DataUtil
import io
# final_test_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/result_L.txt'
final_test_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/result/20160706/data_finalTest_L_result.csv'
output_file_path = 'result/TaskA_all_testdata_15000_ood.csv'


data_util = DataUtil()
final_test_data = data_util.load_data(final_test_file_path)
final_test_data = final_test_data[final_test_data['PREDICT']!= 'UNKOWN']
final_test_data = final_test_data[final_test_data['PREDICT']!= 'other']
final_test_data = final_test_data[final_test_data['PREDICT']!= 'INCONSISTENT']
final_test_data['ID'] = final_test_data['ID'].astype(dtype=int)
final_test_data = final_test_data[['ID','PREDICT']]
# print(final_test_data.head())
# print(final_test_data['PREDICT'].unique)
# print(final_test_data.shape)
# temp = data_util.load_data(('/home/jdwang/PycharmProjects/weiboStanceDetection/data_processing/result/null_data.csv'))
# print(temp.head())
id = [16,20,22,35,67,68,74,78,93,94,101,105,118,123,133,141,151,154,160,161,162,163,168,182,190,202,223,227,238,240,241,247,248,250,253,264,269,270,271,277,282,292,294,298,302,304,312,324,326,331,335,339,356,376,382,389,393,400,401,405,413,415,430,434,436,452,468,484,490,493,494,495,497,507,510,528,537,538,540,553,554,566,569,570,571,581,606,609,612,615,616,617,625,631,635,637,641,648,649,658,668,672,679,687,691,707,710,721,729,733,735,741,759,760,762,763,770,773,774,776,781,807,812,819,820,837,843,852,866,870,882,888,896,914,916,917,919,921,924,929,934,936,945,951,955,965,974,975,982,984,996,1002,1005,1008,1038,1040,1047,1064,1069,1093,1101,1111,1130,1131,1135,1151,1155,1158,1166,1172,1173,1179,1183,1184,1187,1197,1198,1201,1207,1208,1215,1227,1231,1235,1236,1237,1248,1250,1253,1254,1264,1268,1286,1299,1309,1310,1316,1352,1356,1366,1370,1372,1374,1394,1420,1421,1432,1433,1435,1452,1455,1463,1467,1470,1476,1479,1496,1509,1511,1515,1519,1521,1524,1528,1541,1546,1550,1552,1553,1562,1571,1589,1599,1602,1609,1615,1624,1632,1638,1641,1644,1650,1651,1656,1662,1664,1672,1686,1695,1697,1699,1713,1722,1736,1740,1745,1751,1755,1764,1767,1770,1773,1777,1778,1779,1786,1792,1794,1799,1813,1820,1821,1826,1829,1838,1841,1857,1881,1893,1903,1907,1923,1947,1959,1973,1992,2015,2022,2029,2032,2039,2054,2056,2063,2067,2074,2084,2085,2094,2099,2101,2107,2109,2123,2124,2141,2156,2158,2162,2178,2180,2181,2183,2186,2197,2198,2201,2205,2206,2210,2214,2216,2217,2240,2242,2248,2252,2255,2258,2263,2269,2278,2289,2302,2318,2320,2325,2330,2336,2341,2348,2351,2363,2367,2372,2378,2382,2392,2408,2416,2421,2422,2431,2433,2438,2443,2459,2461,2464,2465,2469,2470,2481,2484,2486,2490,2499,2502,2514,2515,2517,2525,2531,2541,2545,2547,2559,2561,2564,2567,2570,2571,2573,2574,2583,2584,2593,2594,2596,2597,2599,2605,2610,2611,2612,2614,2623,2625,2627,2634,2640,2641,2642,2648,2649,2658,2668,2679,2692,2697,2699,2700,2701,2723,2725,2739,2745,2747,2759,2768,2771,2778,2779,2797,2799,2805,2815,2819,2824,2831,2836,2837,2851,2856,2868,2871,2886,2900,2910,2917,2919,2923,2927,2928,2937,2959,2960,2970,2976,2978,2979,2983,2994,2998,9095,9657,9787,9795,9998,10221,10228,10648,10715,11037,11129,11199,11306,11353,11387,11589,454,16]
print(id)
print(sum(final_test_data['ID'].apply(lambda x :x not in id)))
quit()
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
                print(id)
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
                print(id)
                # print(id)
        else:
            fout.write(u'%s\t%s\t%s\t%s\n' % (id, target, text, stance))

print(counter1)
print(counter2)
quit()