# -*- coding: utf-8 -*-
#%%
import sys
sys.path.append("D:\\标签图谱\\标签关系\\recommendation\\Code\\")

import pickle
from imp import reload
import data_generater
reload(data_generater)

#%%
new_file = "../Data/Input/Tag_graph/company_tag_data_concept"
old_file = "../Data/Input/Tag_graph/company_tag_data_non_concept"
comp_ctag_table_all_infos, comp_ctag_table, comp_nctag_table = data_generater.comp_tag(new_file, old_file)

#%%
a, b, c = data_generater.data_aggregater(comp_ctag_table, comp_nctag_table)

#%%
test = pickle.load(open("../Data/Output/recommendation/tag_dict.pkl", "rb"))
test