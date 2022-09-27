#%%
import pandas as pd
import numpy as np
import datetime
import os

goal = [i for i in os.listdir() if i.startswith(
    "车场信息_上海停车_总车位_allMerged")]
goal
# %%
xing_list = ["C", "L", "Z"]

merge_type1_list = [i for i in os.listdir(
    '.') if i.startswith("车场信息_上海停车_总车位_allMerged_type1_2021")]
merge_type2_list = [i for i in os.listdir(
    '.') if i.startswith("车场信息_上海停车_总车位_allMerged_type2_2021")]

merge_type1_time_list = set([i[i.find(".") + 1:i.find("-")]
                             for i in merge_type1_list])
merge_type2_time_list = set([i[i.find(".") + 1:i.find("-")]
                             for i in merge_type2_list])
print(len(merge_type1_list), len(merge_type2_list))
print(merge_type1_time_list, merge_type2_time_list)
# %%
basic_all_merge_byDay_path = [
    "final车场信息_上海停车_总车位_allMerged_type1_2021.", "final车场信息_上海停车_总车位_allMerged_type2_2021."]


def getMerged(merge_name_list, merge_time_list, basic_all_merge_byDay_path, type):
    for i in merge_time_list:
        base_path = basic_all_merge_byDay_path + i + ".csv"
        remain = [k for k in merge_name_list if k.startswith(
            "车场信息_上海停车_总车位_allMerged_type{type}_2021.{a}".format(type=type, a=i))]
        for j in remain:
            if base_path in os.listdir('.'):
                data = pd.read_csv(j)
                data.to_csv(base_path, header=False, index=False, mode='a+')
            else:
                data = pd.read_csv(j)
                data.to_csv(base_path, index=False)
        
        if (type==1):
            data = pd.read_csv(base_path)
            data["TOTAL_BERTH"] = data["BERTH"] / data["PERCENTAGE"]
            # data["TOTAL_BERTH"][data["TOTAL_BERTH"].notnull()] = data["TOTAL_BERTH"][data["TOTAL_BERTH"].notnull()].apply(
            #     lambda x: round(x))
            data.to_csv(base_path, index=False)
        else:
           data = pd.read_csv(base_path)
           data["PERCENTAGE"] = data["BERTH"] / data["TOTAL_BERTH"]
           data["PERCENTAGE"] = data["PERCENTAGE"].apply(lambda x: round(x, 2))
           data.to_csv(base_path, index=False)

# %%
getMerged(merge_type1_list, merge_type1_time_list, basic_all_merge_byDay_path[0], type=1)
# %%
getMerged(merge_type2_list, merge_type2_time_list, basic_all_merge_byDay_path[1], type=2)

# %%
