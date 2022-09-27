#%%
import requests
import pandas as pd
import json
import time
import datetime as d

#%%
index = ["parkLocation", "PAY_TYPE",
         "SERVER_TYPE", "RATE_DESCRIBE", "PARKING_ID", "PARKING_NAME"]
graph = pd.DataFrame(columns=index)
# graph.to_csv("1.csv")
parkID_list = []
#parkId_Dict = {"PARKING_ID": ""}
contents = []

url = r"https://parkapp.jtcx.sh.cn/shcx/app/parkDetailInfo"


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

payload = {"parkingId": ""}

#%%
def getParkingIdList(path, type):
    data = pd.read_csv(path)
    li = data["PARKING_ID2"].values
    li = list(li)
    li = set([i[:i.find("_")] for i in li])
    with open("ParkingId_type{type}.txt".format(type=type), mode="w") as f:
        for i in li:
            f.write(i)
            f.write("\n")


getParkingIdList(
    "处理/final车场信息_上海停车_总车位_allMerged_timeAggregated_type1.csv", type=1)
getParkingIdList(
    "处理/final车场信息_上海停车_总车位_allMerged_timeAggregated_type2.csv", type=2)

#%%
counter = 0
with open("ParkingId_type1.txt", 'r') as f:
    for i in f.readlines():
        i = i.replace("\n", "")
        name = i
        payload["parkingId"] = name
        print("payload: ", payload)
        # time.sleep(1)
        try:
            counter += 1
            res = requests.post(url=url, headers=headers,
                                data=json.dumps(payload), timeout=6).json()

            #print("res: ", res)
            values = res["data"]
            #print("values: ", values)
            if (type(values).__name__ == "dict"):
                for i in index:
                    if (i != "parkLocation"):
                        #print("i: ", i)
                        contents.append(values[i])
                    else:
                        #print("i: ", i)
                        print("values[i]: ", values[i][0]["IMGS"][0])
                        contents.append(values[i][0]["IMGS"][0])
                #print("contents: ", contents)
                graph.loc[len(graph)] = contents
                # print("graph: ", graph)
                contents = []
            else:
                # print(counter)
                # print("values: ", values)
                # print("values_type: ", type(values))
                # print("contents: ", contents)
                # graph.loc[len(graph)] = contents
                contents = []

        except requests.exceptions.ReadTimeout:
            print("Time Out!")

    graph.to_csv(r"\数据\路内停车详细信息_" +
                 d.datetime.now().strftime("%Y.%m.%d-%H-%M") + ".csv", index=False)
    print(graph)

    # with open(r"D:\学科\比赛\计算机设计大赛\代码\爬虫\file5_0_1400_hang.json", 'a+') as f:
    #    f.write(str(res))
    #    #json.dump(res, f, indent=4)



#%%
counter = 0
index2 = ["IMGS", "RATE_DESCRIBE", "ROAD_ID", "ROAD_NAME", "ROAD_PAY_TYPE", "SERVER_TYPE"]
berth_graph = pd.DataFrame(columns=index2)
url2 = r"https://parkapp.jtcx.sh.cn/shcx/app/roadDetailInfo"
payload2 = {"roadId": ""}
with open("ParkingId_type2.txt", 'r') as f:
    for i in f.readlines():
        name = i
        payload2["roadId"] = name
        # time.sleep(1)
        try:
            counter += 1
            res = requests.post(url=url2, headers=headers,
                                data=json.dumps(payload2), timeout=6).json()

            print(res)
            values = res["data"]
            if (type(values).__name__ == "dict"):
                for i in index:
                    if (i != "IMGS"):
                        contents.append(values[i])
                    else:
                        contents.append(values[i][0])
                berth_graph.loc[len(graph)] = contents
                contents = []
            else:
                # berth_graph.loc[len(berth_graph)] = contents
                contents = []

        except requests.exceptions.ReadTimeout:
            print("Time Out!")

    berth_graph.to_csv(r"\数据\路外停车详细信息_" +
                       d.datetime.now().strftime("%Y.%m.%d-%H-%M") + ".csv", index=False)
    print(berth_graph)

# %%
#从爬取的链接中获得图片
img_file = pd.read_csv("数据/路内停车详细信息_2021.04.05-11-21.csv")
header = {
    'Accept': 'text/html, application/xhtml+xml, application/xml;q = 0.9, image/webp, image/apng, */*;q = 0.8, application/signed-exchange;v = b3;q = 0.9',
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN, zh;q = 0.9, en;q = 0.8, en-GB;q = 0.7, en-US;q = 0.6",
    "Cache-Control": "max-age = 0",
    "Host": "180.166.5.202: 32001",
    "If-Modified-Since": "Sun, 18 Oct 2020 09: 14: 13 GMT",
    'If-None-Match': 'W/"207924-1603012453000"',
    "Proxy-Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0 Win64x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.68",
}
def getImg1(line):
    url = line["parkLocation"]
    r = requests.get(url=url, headers=header)
    if r.status_code == 200:
        print (url + "   success")
    content = r.content
    img_path = line["PARKING_NAME"]+".png"
    try:
        with open("图片2\\{img_path}".format(img_path=img_path), 'wb') as fp:
            fp.write(content)
    except:
        print(line["parkLocation"], line["PARKING_NAME"])

# %%
img_file.apply(getImg1, axis=1)
# %%

# %%
