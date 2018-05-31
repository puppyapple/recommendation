import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import reduce
from itertools import product
from Code import data_generator, data_calculator, comp_property
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


comp_infos = ctag_ctag = ctag_nctag = nctag_nctag = ctag_position = comp_id_name_dict = 0

comp_tags_all = "../Data/Output/recommendation/comp_tags_all.pkl"
ctag_ctag = "../Data/Output/recommendation/ctag_ctag.pkl"
ctag_nctag = "../Data/Output/recommendation/ctag_nctag.pkl"
nctag_nctag = "../Data/Output/recommendation/nctag_nctag.pkl"
concept_tree_property = "../Data/Output/recommendation/concept_tree_property.pkl"
ctag_position = "../Data/Output/recommendation/ctag_position.pkl"
comp_id_name_dict = "../Data/Output/recommendation/comp_id_name_dict.pkl"

def all_inputs_generator(new_file, old_file):
    comp_ctag_table, comp_nctag_table, comp_ctag_table_all_infos = data_generator.comp_tag(new_file, old_file)
    print("---Raw data preparation finished---")
    ctag_comps_aggregated, nctag_comps_aggregated, comp_total_num = data_generator.data_aggregator(comp_ctag_table, comp_nctag_table, recalculate=True)
    print("---Data aggregation finished---")
    nctag_idf = data_calculator.nctag_idf(nctag_comps_aggregated, comp_total_num)
    print("---Fake idf of nctag calculation finished---")
    data_calculator.ctag_relation(ctag_comps_aggregated)
    print("---Ctag-ctag relation calculation finished---")
    data_calculator.ctag_nctag_relation(ctag_comps_aggregated, nctag_comps_aggregated, nctag_idf)
    print("---Ctag-nctag relation calculation finished---")
    data_calculator.nctag_nctag(nctag_comps_aggregated, nctag_idf)
    print("---Ntag-nctag relation calculation finished---")
    comp_property.concept_tree_property(comp_ctag_table_all_infos)
    print("---Ctag tree position information preparation finished---")
    return 0


def data_loader(comp_tags_all=comp_tags_all, ctag_ctag=ctag_ctag, ctag_nctag=ctag_nctag, nctag_nctag=nctag_nctag, 
                concept_tree_property=concept_tree_property, ctag_position=ctag_position, comp_id_name_dict=comp_id_name_dict):
    comp_tags_all = pickle.load(open(comp_tags_all, "rb"))
    ctag_ctag = pickle.load(open(ctag_ctag, "rb"))
    ctag_nctag = pickle.load(open(ctag_nctag, "rb"))
    nctag_nctag = pickle.load(open(nctag_nctag, "rb"))
    concept_tree_property = pickle.load(open(concept_tree_property, "rb"))
    ctag_position = pickle.load(open(ctag_position, "rb"))
    comp_id_name_dict = pickle.load(open(comp_id_name_dict, "rb"))
    
    comp_tags_all_df = pd.DataFrame(list(comp_tags_all.items()))
    # comp_tags_all_df.columns = ["comp_id", "tags_infos_dict"]   
    concept_tree_property_df = pd.DataFrame(list(concept_tree_property.items()))
    # concept_tree_property_df.columns = ["comp_id", "concept_tree_property"]
    comp_infos = pd.concat([comp_tags_all_df, concept_tree_property_df]).groupby(0).agg(lambda x: reduce(lambda a, b: {**a, **b} ,x)).reset_index()
    comp_infos.columns = ["comp_id", "comp_property_dict"]
    return (comp_infos, ctag_ctag, ctag_nctag, nctag_nctag, ctag_position, comp_id_name_dict)

def cal_tag_cartesian(tag_set1, tag_set2, value_dict):
    if tag_set1 == 0 or tag_set2 == 0:
        return 0
    else:
        pair_list = list(product(tag_set1, tag_set2))
        value_sum = sum([value_dict.get(t[0] + "-" + t[1], 0) for t in pair_list])
        # print(value_sum)
        return value_sum
    
def cal_tags_link(comp_info1, comp_info2, ctag_ctag, ctag_nctag, nctag_nctag):
    ctags1 = comp_info1.get("ctags", {})
    ctags2 = comp_info2.get("ctags", {})
    nctags1 = comp_info1.get("nctags", {})
    nctags2 = comp_info2.get("nctags", {})
    num_ctags1 = len(ctags1)
    num_ctags2 = len(ctags2)
    num_nctags1 = len(nctags1)
    num_nctags2 = len(nctags2)
    num_all1 = num_ctags1 + num_nctags1
    num_all2 = num_ctags2 + num_nctags2
    
    coef1 = 1/np.sqrt(1 + (num_ctags1 - num_ctags2)**2)
    coef2 = 1/np.sqrt(1 + (num_nctags1 - num_nctags2)**2)
    coef3 = data_calculator.final_count(nctags1, nctags2)
    
    v1 = coef1 * cal_tag_cartesian(ctags1, ctags2, ctag_ctag)
    v2 = cal_tag_cartesian(ctags1, nctags2, ctag_nctag) + cal_tag_cartesian(ctags2, nctags1, ctag_nctag)
    v3 = coef3 * cal_tag_cartesian(nctags1, nctags2, nctag_nctag)
    return (v1, v2, v3)
    
def cal_part(target_comp_info, comp_infos, ctag_ctag, ctag_nctag, nctag_nctag, weights=(0.6, 0.2, 0.2)):
    print("start")
    # target_comp_info = list(comp_infos[comp_infos.comp_id == comp_id].comp_property_dict)[0]
    three_values = np.array(list(comp_infos.comp_property_dict.apply(lambda x: cal_tags_link(target_comp_info, x, ctag_ctag, ctag_nctag, nctag_nctag))))
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(three_values)
    tmp = comp_infos.copy()
    tmp["sim_value"] = (scaler.transform(three_values) *  weights).sum(axis=1)
    return tmp

def cal_simple(target_comp_info, part):
    return cal_part(target_comp_info, part, ctag_ctag, ctag_nctag, nctag_nctag)

def cal_company_dis(target_comp_info, part,  weights=(0.6, 0.2, 0.2)):
    print("start")
    three_value_list = list(comp_infos.comp_property_dict.apply(lambda x: cal_tags_link(target_comp_info, x, ctag_ctag, ctag_nctag, nctag_nctag)))
    part["three_values"] = three_value_list
    return part

def multi_process_rank(comp_id, weights=(0.6, 0.2, 0.2), response_num=100, process_num=8):
    target_comp_info = list(comp_infos[comp_infos.comp_id == comp_id].comp_property_dict)[0]
    result_list = []
    split_comp_infos = np.array_split(comp_infos, process_num)
    pool = mp.Pool()
    for i in range(0, process_num):
        result_list.append(pool.apply_async(cal_company_dis, (target_comp_info, split_comp_infos[i], )))
    pool.close()
    pool.join()
    result_merged = pd.concat([r.get() for r in result_list])
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(result_merged.three_values)
    result_merged["sim_value"] = (scaler.transform(three_values) *  weights).sum(axis=1)
    result_sorted = result_merged.sort_values(by="sim_value", ascending=False)[:response_num].copy()
    return result_sorted
    