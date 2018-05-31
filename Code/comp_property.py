# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle

def concept_tree_property(comp_ctag_table_all_infos):   
    tag_code_dict = pd.DataFrame.from_dict(pickle.load(open("../Data/Output/recommendation/tag_dict.pkl", "rb")), orient="index").reset_index()
    tag_code_dict.columns = ["label_name", "tag_code"]
    com_ctag_with_code = comp_ctag_table_all_infos.merge(tag_code_dict, how="left", left_on="label_name", right_on="label_name")
    # 每个公司的概念最底层概念标签列表：判断上下游、同链条、同标签
    comp_bottom_ctag = com_ctag_with_code.groupby(["comp_id", "label_type_num"]).apply(lambda x: x[x.label_type == x.label_type.max()])[["comp_id", "tag_code", "comp_full_name"]].drop_duplicates().reset_index(drop=True)
    comp_bottom_ctag.columns = ["comp_id", "bottom_ctag", "comp_full_name"]
    comp_bottom_ctag = comp_bottom_ctag.groupby("comp_id").agg({"bottom_ctag": lambda x: set(x)}).reset_index()
    # 每个公司的顶级概念标签列表：判断同产业“树”
    comp_top_ctag = com_ctag_with_code.groupby(["comp_id", "label_type_num"]).apply(lambda x: x[x.label_type == x.label_type.min()])[["comp_id", "tag_code", "comp_full_name"]].drop_duplicates().reset_index(drop=True)
    comp_top_ctag.columns = ["comp_id", "top_ctag", "comp_full_name"]
    comp_top_ctag = comp_top_ctag.groupby("comp_id").agg({"top_ctag": lambda x: set(x)}).reset_index()
    concept_tree_property = comp_top_ctag.merge(comp_bottom_ctag, how="left", left_on="comp_id", right_on="comp_id")
    concept_tree_property.index = concept_tree_property.comp_id
    concept_tree_property.drop(["comp_id"], axis=1, inplace=True)
    concept_tree_property_dict = concept_tree_property.to_dict(orient='index')
    concept_tree_property_dict_file_name = "../Data/Output/recommendation/concept_tree_property.pkl"
    concept_tree_property_dict_file = open("../Data/Output/recommendation/concept_tree_property.pkl", "wb")
    pickle.dump(concept_tree_property_dict, concept_tree_property_dict_file)
    concept_tree_property_dict_file.close()
    return concept_tree_property