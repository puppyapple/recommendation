# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as numpy

def comp_tag(new_file, old_file):
    '''
    根据输入的公司概念和非概念标记源数据，分别得到完整的公司-概念标签、公司-非概念标签一对一表
    '''
    data_raw_ctag = pd.read_csv(new_file, sep='\t', dtype={"comp_id":str})
    data_raw_ctag.dropna(subset=["comp_id", "label_name"], inplace=True)
    cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
    data_raw_ctag = data_raw_ctag[data_raw_ctag.label_name != ''][cols].copy()

    # comp_ctag_table_all_infos 是读取文件后带非概念标签的全部信息
    comp_ctag_table_all_infos = data_raw_ctag[data_raw_ctag.classify_id != 4].reset_index(drop=True)
    # 公司-标签表只取部分字段
    comp_ctag_table = comp_ctag_table_all_infos[["comp_id", "label_name"]].copy()
    # 数字1表示该行记录里的标签为概念标签
    comp_ctag_table["type"] = 1
    data_raw_nctag_p1 = data_raw_ctag[data_raw_ctag.classify_id == 4][["comp_id", "label_name"]].copy()

    # 读取旧版数据
    data_raw_nctag = pd.read_csv(old_file, sep='\t', dtype={"comp_id": str, "comp_full_name": str, "key_word": str})[["comp_id", "key_word"]]
    data_raw_nctag.dropna(subset=["comp_id", "key_word"], inplace=True)
    data_raw_nctag.columns = ["comp_id", "label_name"]
    data_raw_nctag_p2 = data_raw_nctag[data_raw_nctag.key_word != ""].copy()
    
    #
    data_raw_nctag_merged = pd.concat([data_raw_nctag_p1, data_raw_nctag_p2]).drop_duplicates().reset_index(drop=True)
    tuples = data_raw_nctag_merged.apply(lambda x: [(x[0], t) for t in x[1].split(",") if t != ""], axis=1)
    flatted = [y for x in tuples for y in x]
    comp_nctag = pd.DataFrame(flatted, columns=["comp_id", "comp_full_name", "tag"]).drop_duplicates()

    return (comp_ctag_table_all_infos, comp_tag_table)