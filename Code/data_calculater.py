# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
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

    # 过滤同链标签
    

    ctag_ctag.link_value = ctag_ctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    ctag_ctag.link_value = simple_minmax(ctag_ctag.link_value)
    ctag_ctag = ctag_ctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    return ctag_ctag[["tag_uuid_x", "tag_uuid_y", "link_value"]]

# 概念标签和非概念标签关系计算
def ctag_nctag_relation(ctag_comps_aggregated, nctag_comps_aggregated, nctag_idf):
    ctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated = nctag_comps_aggregated.merge(nctag_idf, how="left", left_on="tag_uuid", right_on="tag_uuid")
    ctag_nctag = ctag_comps_aggregated.merge(ctag_comps_aggregated, on="key")
    ctag_nctag["link_value"] = ctag_nctag[["comp_int_id_x", "comp_int_id_y", "idf"]].apply(lambda x: x[2]*final_count(x[0], x[1]), axis=1)
    ctag_nctag = ctag_nctag[ctag_nctag.link_value != 0].copy()
    ctag_nctag.link_value = ctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    ctag_nctag.link_value = simple_minmax(ctag_nctag.link_value)
    ctag_nctag = ctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    return ctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]]

# 非概念标签两两关系计算
def nctag_nctag(nctag_comps_aggregated, nctag_idf):
    nctag_comps_aggregated["key"] = 1
    nctag_nctag = nctag_comps_aggregated.merge(nctag_comps_aggregated, on="key")
    nctag_nctag["link_value"] = nctag_nctag[["comp_int_id_x", "comp_int_id_y", "idf_x", "idf_y"]] \
        .apply(lambda x: x[2]*x[3]*final_count(x[0], x[1]), axis=1)
    nctag_nctag = nctag_nctag[nctag_nctag.link_value != 0].copy()
    nctag_nctag.link_value = nctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    nctag_nctag.link_value = simple_minmax(nctag_nctag.link_value)
    nctag_nctag = nctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    return nctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]]