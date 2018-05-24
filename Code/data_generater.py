# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as numpy
import uuid
import pickle
nctag_filter_num = 50


def final_count(l1, l2):
    return len(l1.intersection(l2))/len(l1.union(l2))

def comp_tag(new_file, old_file):
    '''
    根据输入的公司概念和非概念标记源数据，分别得到完整的公司-概念标签、公司-非概念标签
    '''
    data_raw_new = pd.read_csv(new_file, sep='\t', dtype={"comp_id":str})
    data_raw_new.dropna(subset=["comp_id", "label_name"], inplace=True)
    cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
    data_raw_new = data_raw_new[data_raw_new.label_name != ''][cols].copy()

    # comp_ctag_table_all_infos 是读取文件后带非概念标签的全部信息
    comp_ctag_table_all_infos = data_raw_new[data_raw_new.classify_id != 4].reset_index(drop=True)
    comp_ctag_table = comp_ctag_table_all_infos[["comp_id", "label_name"]].reset_index(drop=True)
    # print(comp_ctag_table)
    # 概念标签全集列表，加上一列1作为标记用
    ctag_list = comp_ctag_table_all_infos.label_name.drop_duplicates().reset_index()
    ctag_list.rename(index=str, columns={"index": "ctag_mark"}, inplace=True)
    ctag_list.ctag_mark = ctag_list.ctag_mark.apply(lambda x: 1)

    # 新系统下结果中的公司-非概念标签
    data_raw_nctag_p1 = data_raw_new[data_raw_new.classify_id == 4][["comp_id", "label_name"]].copy()

    # 读取旧版数据，只取其中的非概念标签（概念标签无法确定其层级和产业链（复用））
    data_raw_old = pd.read_csv(old_file, sep='\t', dtype={"comp_id": str, "comp_full_name": str, "key_word": str}) \
        [["comp_id", "key_word"]]
    data_raw_old.dropna(subset=["comp_id", "key_word"], inplace=True)
    data_raw_old.columns = ["comp_id", "label_name"]
    data_raw_old = data_raw_old[data_raw_old.label_name != ""].copy()

    # 新版的非概念标签和旧版整体数据拼接后进行split和flatten
    data_to_flatten = pd.concat([data_raw_old, data_raw_nctag_p1])
    tuples = data_to_flatten.apply(lambda x: [(x[0], t) for t in x[1].split(",") if t != ""], axis=1)
    flatted = [y for x in tuples for y in x]
    data_raw_nctag_flatted = pd.DataFrame(flatted, columns=["comp_id", "label_name"]).drop_duplicates()
    data_raw_nctag_with_mark = data_raw_nctag_flatted.merge(ctag_list, how="left", left_on="label_name", right_on="label_name")

    # 取没有概念标记的作为非概念标签的全集
    comp_nctag_table = data_raw_nctag_with_mark[data_raw_nctag_with_mark.ctag_mark != 1] \
        .drop(["ctag_mark"], axis=1).reset_index(drop=True)

    return (comp_ctag_table_all_infos, comp_ctag_table, comp_nctag_table)


def data_aggregater(comp_ctag_table, comp_nctag_table):
    comp_tag_table_all = pd.concat([comp_ctag_table, comp_nctag_table])
    # 为每一个公司赋予一个整数ID，以减小之后的计算量
    comp_id_dict = comp_tag_table_all["comp_id"].drop_duplicates().reset_index(drop=True)
    comp_id_dict = comp_id_dict.reset_index()
    comp_id_dict.rename(index=str, columns={"index": "comp_int_id"}, inplace=True)
    comp_ctag_table = comp_ctag_table.merge(comp_id_dict, how="left", left_on="comp_id", right_on="comp_id") \
        .drop(["comp_id"], axis=1)
    comp_nctag_table = comp_nctag_table.merge(comp_id_dict, how="left", left_on="comp_id", right_on="comp_id") \
        .drop(["comp_id"], axis=1)

    # 为每一个标签赋予一个UUID
    tag_list = comp_tag_table_all["label_name"].drop_duplicates().reset_index(drop=True)
    tag_list = tag_list.reset_index()
    tag_list.rename(index=str, columns={"index": "tag_uuid"}, inplace=True)
    tag_list.tag_uuid = tag_list.label_name.apply(lambda x: uuid.uuid5(uuid.NAMESPACE_URL, x).hex)

    comp_ctag_table = comp_ctag_table.merge(tag_list, how="left", left_on="label_name", right_on="label_name") \
        .drop(["label_name"], axis=1)
    comp_nctag_table = comp_nctag_table.merge(tag_list, how="left", left_on="label_name", right_on="label_name") \
        .drop(["label_name"], axis=1)

    # 将标签对应的hashcode以字典形式存成二进制文件
    tag_dict = dict(zip(tag_list.label_name, tag_list.tag_uuid))
    tag_dict_file = open("../Data/Output/recommendation/tag_dict.pkl", "wb")
    pickle.dump(tag_dict, tag_dict_file)
    tag_dict_file.close()

    # 将概念非概念标签数据各自按照标签id进行聚合
    ctag_comps_aggregated = comp_ctag_table.groupby("tag_uuid").agg({"comp_int_id": lambda x: set(x)}).reset_index()
    comp_nctag_table["count_comps"] = 1
    nctag_comps_aggregated_all = comp_nctag_table.groupby("tag_uuid") \
        .agg({"comp_int_id": lambda x: set(x), "count_comps": "count"}).reset_index()
    nctag_comps_aggregated = nctag_comps_aggregated_all[nctag_comps_aggregated_all.count_comps >= nctag_filter_num] \
        [["tag_uuid", "comp_int_id"]].reset_index(drop=True)
    return (ctag_comps_aggregated, nctag_comps_aggregated, comp_properties, tag_dict)

