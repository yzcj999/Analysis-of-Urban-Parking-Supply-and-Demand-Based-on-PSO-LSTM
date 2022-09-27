# In[1]:


import pandas as pd
import numpy as np
import datetime
import os


# In[2]:


print(os.listdir('.'))


# In[3]:


merge_type1_list = [i for i in os.listdir(
    '.') if i.startswith("车场信息_上海停车_总车位_merge_type1_2021")]
merge_type2_list = [i for i in os.listdir(
    '.') if i.startswith("车场信息_上海停车_总车位_merge_type2_2021")]

print(len(merge_type1_list), len(merge_type2_list))


# In[4]:


merge_type1_time_list = set([i[i.find(".") + 1:i.find("-")]
                             for i in merge_type1_list])
merge_type2_time_list = set([i[i.find(".") + 1:i.find("-")]
                             for i in merge_type2_list])

print(merge_type1_time_list, merge_type2_time_list)


# In[5]:


max_size_merge_type1_list = [os.stat(i).st_size for i in merge_type1_list]
max_size_merge_type1_file = merge_type1_list[max_size_merge_type1_list.index(
    max(max_size_merge_type1_list))]
print(max_size_merge_type1_file)


# In[6]:


max_size_merge_type2_list = [os.stat(i).st_size for i in merge_type2_list]
max_size_merge_type2_file = merge_type2_list[max_size_merge_type2_list.index(
    max(max_size_merge_type2_list))]
print(max_size_merge_type2_file)


# In[7]:


basic_all_merge_byDay_path = [
    "车场信息_上海停车_总车位_allMerged_type1_2021.", "车场信息_上海停车_总车位_allMerged_type2_2021."]


def getMerged(merge_name_list, merge_time_list, basic_all_merge_byDay_path, type, xing):

    # 将每个文件加上时间头
    for i in merge_time_list:
        # print(i)
        remain = [k for k in merge_name_list if k.startswith(
            "车场信息_上海停车_总车位_merge_type{type}_2021.{a}".format(type=type, a=i))]
        # print(merge_name_list)
        # print(remain)
        for j in remain:
            # print(j)
            time = j[j.find(".") - 4:j.rfind(".")]
            data = pd.read_csv(j)
            time = time[:time.find("-")] + " " + time[time.find("-") + 1:]
            # print(time[time.find("-")])
            data["starttime"] = [time for i in range(len(data))]
            print(time)
            data["starttime"] = pd.to_datetime(
                data["starttime"], format="%Y.%m.%d %H-%M")
            print(data)
            data.to_csv(j, index=False)

    for i in merge_time_list:
        base_path = basic_all_merge_byDay_path + i + "-" + xing + ".csv"
        remain = [k for k in merge_name_list if k.startswith(
            "车场信息_上海停车_总车位_merge_type{type}_2021.{a}".format(type=type, a=i))]
        for j in remain:
            if base_path in os.listdir('.'):
                data = pd.read_csv(j)
                data.to_csv(base_path, header=False, index=False, mode='a+')
            else:
                data = pd.read_csv(j)
                data.to_csv(base_path, index=False)

    data = pd.read_csv(base_path)
    data.head(30)


# In[8]:


getMerged(merge_type1_list, merge_type1_time_list,
          basic_all_merge_byDay_path=basic_all_merge_byDay_path[0], type=1, xing="Z")


# In[9]:


getMerged(merge_type2_list, merge_type2_time_list,
          basic_all_merge_byDay_path=basic_all_merge_byDay_path[1], type=2, xing="Z")


# In[ ]:
