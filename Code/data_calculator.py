# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def final_count(l1, l2):
    return len(l1.intersection(l2))/len(l1.union(l2))

def simple_minmax(column_target, min_v=0.001, max_v=1):
    target = column_target.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(min_v, max_v))
    scaler.fit(target)
    return scaler.transform(target)

# 考虑分开计算三组关系，因为非概念之间的内存占用可能较大

# 先计算每个非概念标签的伪TF-IDF系数
def nctag_idf(nctag_comps_aggregated, comp_total_num):
    nctag_idf = nctag_comps_aggregated[["tag_uuid"]].copy()
    nctag_idf["idf"] = nctag_comps_aggregated.comp_int_id.apply(lambda x: np.log2(comp_total_num/len(x)))
    nctag_idf.idf = simple_minmax(nctag_idf.idf)
    return nctag_idf

# 概念标签两两关系计算
def ctag_relation(ctag_comps_aggregated):
    ctag_comps_aggregated["key"] = 1
    ctag_ctag = ctag_comps_aggregated.merge(ctag_comps_aggregated, on="key")
    ctag_ctag["link_value"] = ctag_ctag[["comp_int_id_x", "comp_int_id_y"]].apply(lambda x: final_count(x[0], x[1]), axis=1)
    ctag_ctag = ctag_ctag[ctag_ctag.link_value != 0].copy()

    # 过滤同链标签（保留相邻标签节点）
    ctag_position = pickle.load(open("../Data/Output/recommendation/ctag_position.pkl", "rb"))
    label_chains_all = pd.DataFrame.from_dict(ctag_position, orient="index").reset_index()
    label_chains_all.columns = ["node_link", "distance"]
    label_chains_all = label_chains_all[label_chains_all.distance > 1]
    # 将相邻节点外的标签对统一标记为1，join后进行剔除
    label_chains_all.distance = 1
    ctag_ctag["node_link"] = ctag_ctag[["tag_uuid_x", "tag_uuid_y"]].apply(lambda x: x[0] + "-" + x[1], axis=1)
    ctag_ctag = ctag_ctag.merge(label_chains_all, how="left", left_on="node_link", right_on="node_link")
    ctag_ctag = ctag_ctag[ctag_ctag.distance != 1].drop(["node_link", "distance"], axis=1)
    
    ctag_ctag.link_value = ctag_ctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    ctag_ctag.link_value = simple_minmax(ctag_ctag.link_value)
    ctag_ctag = ctag_ctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    ctag_ctag["tag_link"] = ctag_ctag.tag_uuid_x + "-" + ctag_ctag.tag_uuid_y
    ctag_ctag_dict = dict(zip(ctag_ctag.tag_link, ctag_ctag.link_value))
    ctag_ctag_file_name = "../Data/Output/recommendation/ctag_ctag.pkl"
    ctag_ctag_file = open(ctag_ctag_file_name, "wb")
    pickle.dump(ctag_ctag_dict, ctag_ctag_file)
    ctag_ctag_file.close()
    # ctag_ctag.to_csv("../Data/Output/recommendation/ctag_ctag_result.csv", index=False, header=False)
    return ctag_ctag

# 概念标签和非概念标签关系计算
def ctag_nctag_relation(ctag_comps_aggregated, nctag_comps_aggregated, nctag_idf):
    ctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated = nctag_comps_aggregated.merge(nctag_idf, how="left", left_on="tag_uuid", right_on="tag_uuid")
    ctag_nctag = nctag_comps_aggregated.merge(ctag_comps_aggregated, on="key")
    ctag_nctag["link_value"] = ctag_nctag[["comp_int_id_x", "comp_int_id_y", "idf"]].apply(lambda x: x[2]*final_count(x[0], x[1]), axis=1)
    ctag_nctag = ctag_nctag[ctag_nctag.link_value != 0].copy()
    ctag_nctag.link_value = ctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    ctag_nctag.link_value = simple_minmax(ctag_nctag.link_value)
    ctag_nctag = ctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    ctag_nctag["tag_link"] = ctag_nctag.tag_uuid_x + "-" + ctag_nctag.tag_uuid_y
    ctag_nctag_dict = dict(zip(ctag_nctag.tag_link, ctag_nctag.link_value))
    ctag_nctag_file_name = "../Data/Output/recommendation/ctag_nctag.pkl"
    ctag_nctag_file = open(ctag_nctag_file_name, "wb")
    pickle.dump(ctag_nctag_dict, ctag_nctag_file)
    ctag_nctag_file.close()
    # ctag_nctag.to_csv("../Data/Output/recommendation/ctag_nctag_result.csv", index=False, header=False)
    return ctag_nctag

# 非概念标签两两关系计算
def nctag_nctag(nctag_comps_aggregated, nctag_idf):
    nctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated = nctag_comps_aggregated.merge(nctag_idf, how="left", left_on="tag_uuid", right_on="tag_uuid")
    nctag_nctag = nctag_comps_aggregated.merge(nctag_comps_aggregated, on="key")
    record_len = len(nctag_nctag)
    interval_size = record_len // 10
    i = 0
    while interval_size*i < record_len:
        start_time = datetime.datetime.now()
        print("### start part %d at %s ###" % (i, start_time.strftime('%H:%M:%S')))
        tmp = nctag_nctag[interval_size*i:min(interval_size * (i + 1), record_len)]
        tmp["link_value"] = tmp[["comp_int_id_x", "comp_int_id_y", "idf_x", "idf_y"]].copy().apply(lambda x: x[2] * x[3] * final_count(x[0], x[1]), axis=1)
        result_part = tmp[tmp.link_value != 0][["tag_uuid_x", "tag_uuid_y", "link_value"]]
        result_part.to_csv("../Data/Output/recommendation/temp_result/part_result_%d.relations" % i, index=False, header=None)
        end_time = datetime.datetime.now()
        print("### Part %d finished at %s (time used: %.3f seconds) ###" % (i, end_time.strftime('%H:%M:%S'), (end_time - start_time).total_seconds()))
        i += 1
    print("calculation done")
    os.system("cat ../Data/Output/recommendation/temp_result/part_result_* > ../Data/Output/recommendation/temp_result/nctag_nctag_result_all")
    nctag_nctag = pd.read_csv("../Data/Output/recommendation/temp_result/nctag_nctag_result_all", header=None)
    nctag_nctag.columns = ["tag1", "tag2", "link_value"]
    nctag_nctag.link_value = nctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    nctag_nctag.link_value = simple_minmax(nctag_nctag.link_value)
    nctag_nctag = nctag_nctag[["tag1", "tag2", "link_value"]].drop_duplicates().copy()
    nctag_nctag["tag_link"] = nctag_nctag.tag1 + "-" + nctag_nctag.tag2
    nctag_nctag_dict = dict(zip(nctag_nctag.tag_link, nctag_nctag.link_value))
    nctag_nctag_file_name = "../Data/Output/recommendation/nctag_nctag.pkl"
    nctag_nctag_file = open(nctag_nctag_file_name, "wb")
    pickle.dump(nctag_nctag_dict, nctag_nctag_file)
    nctag_nctag_file.close()
    # nctag_nctag.to_csv("../Data/Output/recommendation/nctag_nctag_result.csv", index=False, header=False)
    return nctag_nctag

# 合并三种标签关系的结果并乘以权重后以tag_link—link_value的形式储存起来
def result_merge(ctag_ctag_path, ctag_nctag_path, nctag_nctag_path, merged_path):
    ctag_ctag = pd.read_csv(ctag_ctag_path, header=None)
    # ctag_ctag.columns = ["tag1", "tag2", "link_value"]
    # ctag_ctag.link_value = ctag_ctag.link_value * weight.get("ctag_ctag")
    
    ctag_nctag = pd.read_csv(ctag_nctag_path, header=None)
    # ctag_nctag.columns = ["tag1", "tag2", "link_value"]
    # ctag_nctag.link_value = ctag_nctag.link_value * weight.get("ctag_nctag")
    
    nctag_nctag = pd.read_csv(nctag_nctag_path, header=None)
    # nctag_nctag.columns = ["tag1", "tag2", "link_value"]
    # nctag_nctag.link_value = nctag_nctag.link_value * weight.get("nctag_nctag")
    
    result_all = pd.concat([ctag_ctag, ctag_nctag, nctag_nctag]).drop_duplicates()
    result_all.columns = ["tag1", "tag2", "link_value"]
    result_all["tag_link"] = result_all.tag1 + "-" + result_all.tag2
    result_all = result_all[["tag_link", "link_value"]].copy()
    
    result_all_dict = dict(zip(result_all.tag_link, result_all.link_value))
    tag_relation_all_file_name = "../Data/Output/recommendation/tag_relation_all.pkl"
    tag_relation_all_file = open(tag_relation_all_file_name, "wb")
    pickle.dump(result_all_dict, tag_relation_all_file)
    tag_relation_all_file.close()
    return result_all