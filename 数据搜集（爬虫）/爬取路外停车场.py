import requests
import pandas as pd
import json
import time
import datetime as d


index = ["ADDRESS", "BUSY_STATUS", "LATITUDE",
         "LONGITUDE", "P_TYPE", "PARKING_ID", "PARKING_NAME"]
graph = pd.DataFrame(columns=index)
# graph.to_csv("1.csv")
parkID_list = []
#parkId_Dict = {"PARKING_ID": ""}
contents = []

url = r"https://parkapp.jtcx.sh.cn/shcx/app/listBasePark"


headers = {
    "Host": "parkapp.jtcx.sh.cn",
    "Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Html5Plus/1.0 (Immersed/44)",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-cn",
    "authToken": "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MTc2MjE5NzUsInN1YiI6IjZhYzAwMjE5ODI4MTRhNzRiYzZhZTM1ZWMzMTFkM2Q3NjMxMjAyMTAzMjkxOTI2MTUiLCJpYXQiOjE2MTcwMTcxNzV9.ybO_VNIXc9oWnbjI-XIvVxH_eLOpSpbN5wiK5144mi8",
    "tokenKey": "1617017175886",
    "Content-Type": "application/json",
    "Origin": "null",
    #"Content-Length": "151",
    "sign": "",
}

# 注意pageSize表示的是返回多少的数据
# P_type 表示的是停车场的信息  0--全部  1--路外  2--路内
payload = {"pageNum": 0, "pageSize": 25, "params": {"latitude": 31.233462, "longitude": 121.492156,
                                                    "currentLatitude": 34.781425, "currentLongitude": 113.663457, "P_TYPE": 1}}

counter = 0

with open("选取的地点.txt", 'r') as f:
    for i in f.readlines():
        i = i.replace("\n", "")
        longi, lati = i.split(',')
        #longi = longi.replace('.', '')
        #lati = lati.replace('.', '')

        payload["params"]["latitude"] = float(lati)
        payload["params"]["longitude"] = float(longi)
        # time.sleep(1)
        try:
            counter += 1
            res = requests.post(url=url, headers=headers,
                                data=json.dumps(payload), timeout=6).json()

            # print(res)
            values = res["data"]
            if (type(values).__name__ == "list"):
                for each_json in values:
                    # print(type(each_json))
                    for i in index:
                        contents.append(each_json[i])
                        #print("append done\n")
                    # print(contents)
                    #print( "contents: " ,contents)
                    #new_entry = pd.DataFrame(data=contents, columns=index)
                    #print("new-entry: ", new_entry)
                    #graph = pd.concat([graph, new_entry], axis=0)

                    # 将contents中的PARKING_ID字段提取
                    parkId_Dict = {"PARKING_ID": ""}
                    parkId_Dict["PARKING_ID"] = contents[5]
                    parkID_list.append(parkId_Dict)
                    graph.loc[len(graph)] = contents
                    contents = []
            else:
                print(counter)
                print("values: ", values)
                print("values_type: ", type(values))
                for i in index:
                    contents.append(values[i])

                # print(contents)
                #new_entry = pd.DataFrame(data=contents, columns=index)

                # 将contents中的PARKING_ID字段提取
                parkId_Dict = {"PARKING_ID": ""}
                parkId_Dict["PARKING_ID"] = contents[5]
                parkID_list.append(parkId_Dict)
                graph.loc[len(graph)] = contents
                #graph = pd.concat([graph, new_entry], axis=0)
                contents = []

        except requests.exceptions.ReadTimeout:
            print("Time Out!")

    graph.to_csv(r"\数据\车场信息_上海停车_总车位_type1_" +
                 d.datetime.now().strftime("%Y.%m.%d-%H-%M") + ".csv", index=False)
    print(graph)


berth_url = "https://parkapp.jtcx.sh.cn/shcx/app/getParkBerth"
berth_headers = {
    "Host": "parkapp.jtcx.sh.cn",
    "Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Html5Plus/1.0 (Immersed/44)",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-cn",
    "authToken": "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MTc2MjE5NzUsInN1YiI6IjZhYzAwMjE5ODI4MTRhNzRiYzZhZTM1ZWMzMTFkM2Q3NjMxMjAyMTAzMjkxOTI2MTUiLCJpYXQiOjE2MTcwMTcxNzV9.ybO_VNIXc9oWnbjI-XIvVxH_eLOpSpbN5wiK5144mi8",
    "tokenKey": "1617017175886",
    "Content-Type": "application/json",
    "Origin": "null",
    #"Content-Length": "151",
    "sign": "",
}
data = parkID_list
# print(data)
#data = list(set(data))
berth_index = ["BERTH", "PARKING_ID", "PERCENTAGE"]
berth_graph = pd.DataFrame(columns=berth_index)
contents2 = []

try:
    res = requests.post(url=berth_url, headers=berth_headers,
                        data=json.dumps(data), timeout=6).json()

    # print(res)
    values = res["data"]
    if (type(values).__name__ == "list"):
        #print("len: ", len(values))
        if (len(values) != 0):
            for each_json in values:
                # print(type(each_json))
                for i in berth_index:
                    try:
                        contents2.append(each_json[i])
                    except TypeError:
                        print(each_json)
                    #print("append done\n")
                    # print(contents)
                    #print("contents: ", contents)
                    #new_entry = pd.DataFrame(data=contents, columns=index)
                    #print("new-entry: ", new_entry)
                    #graph = pd.concat([graph, new_entry], axis=0)

                berth_graph.loc[len(berth_graph)] = contents2
                contents2 = []
        else:
            # 表示请求的数据就是全空的
            contents2 = [-1, -1, -1]
            berth_graph.loc[len(berth_graph)] = contents2
            contents2 = []
    else:
        print(values)
        # for i in berth_index:
        #     contents2.append(each_json[i])

        #     # print(contents)
        #     #new_entry = pd.DataFrame(data=contents, columns=index)

        # berth_graph.loc[len(berth_graph)] = contents2
        #graph = pd.concat([graph, new_entry], axis=0)
        contents2 = []

except requests.exceptions.ReadTimeout:
    print("Time Out!")

berth_graph.to_csv(r"\数据\车场信息_上海停车_总车位_berth_type1_" +
                   d.datetime.now().strftime("%Y.%m.%d-%H-%M") + ".csv", index=False)
print(berth_graph)

graph_merge = pd.merge(graph, berth_graph, how="outer")
graph_merge = graph_merge.drop_duplicates()
graph_merge.to_csv(r"\数据\车场信息_上海停车_总车位_merge_type1_" +
                   d.datetime.now().strftime("%Y.%m.%d-%H-%M") + ".csv", index=False)


# 获取停车指数
parking_index_url = "https://parkapp.jtcx.sh.cn/shcx/app/getParkingIndex"
parking_index_data = {}

parking_index_res = requests.post(
    url=parking_index_url, headers=berth_headers, data=parking_index_data).json()
parkindex = pd.DataFrame(parking_index_res["data"])
parkindex["DATA_TIME"] = pd.to_datetime(parkindex["DATA_TIME"])
parkindex.to_csv("all_PARKING_INDEX_DATA.csv",
                 index=False, header=False, mode="a+")
