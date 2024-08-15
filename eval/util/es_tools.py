from multiprocessing import Pool
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import requests
from multiprocessing import Pool
from requests.auth import HTTPBasicAuth
import json
from datetime import datetime
import os
from PIL import Image
import base64


class ES(object):
    def __init__(self, host,http_auth,index_name,mapping):
        self.index_name = index_name
        self.host = host
        self.http_auth = http_auth
        self.es = Elasticsearch(self.host, http_auth=self.http_auth)
        self.mapping = mapping
    def verify_index(self):
        # Elasticsearch地址
        es_url = self.host
        # 用户名和密码
        username = self.http_auth[0]
        password = self.http_auth[1]
        # 发送GET请求，包含身份验证信息
        response = requests.get(f'{es_url}/_cat/indices?v', auth=HTTPBasicAuth(username, password))
        # 输出响应内容（即索引列表）
        print(response.text)

    def create_index(self):
        # 更改为自己es库中需要的embedding大小
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=self.mapping)
        else:
            print("索引已存在,索引mapping如下：")
            mapping = self.es.indices.get_mapping(index=self.index_name)
            print(mapping)

    def delete_index(self):
        self.es.indices.delete(self.index_name)

    def verify_index(self):
        es_url =self.host
        username = self.http_auth[0]
        password = self.http_auth[1]
        response = requests.get(f'{es_url}/_cat/indices?v', auth=HTTPBasicAuth(username, password),timeout=10)
        print(response.text)

    def es_search(self,feature,search_num=5):
        # 要搜索的索引名称和索引中存放的embedding大小
        size =  self.mapping["mappings"]['doc']['properties']['feature']['dim']
        body = {
            "from": 0,
            "size": search_num,
            "query": {
            "dis_max": {
                        "queries": [
                            {
                                "hnsw": {
                                    "feature":{
                                    "size" : size,
                                    "vector": feature #[float(x) for x in text_dict["feature_full"]]
                            }
                            }},
                        ]
                    }
            }
        }
        # 执行查询
        response = None
        while response is None:
            try:
                response = self.es.search(index=self.index_name, body=body)
            except Exception as e:
                response = None
        # print(json.dumps(response, indent=4))
        return response['hits']['hits']
    
    def es_search_id(self,id):
        search_query = {
            "query": {
                "term": {
                    "question_id": id
                }
            }
        }
        # 执行搜索查询
        response = None
        while response is None:
            try:
                response = self.es.search(index=self.index_name, body=search_query)
            except Exception as e:
                response = None
        return response['hits']['hits']

def es_insert(es_tool,item):
    try:
        # 统一位数，防止搜索不到，此处将当前18位小数改为16位小数
        actions=[]
        if type(item[1]) == dict:
            feature = item[1]['feature']
        else:
            feature = item[1]
        feature = [round(x,18) for x in feature]

        action = {'_source': {'feature':feature,'id':item[0]}}
        actions.append(action)
        if len(actions) == 100:
            success, _ = bulk(es_tool.es, actions, index=es_tool.index_name, raise_on_error=True, doc_type="doc")
            actions = []    
        if len(actions) > 0:
            success, _ = bulk(es_tool.es, actions, index=es_tool.index_name, raise_on_error=True, doc_type="doc")
        return success,None
    except Exception as e:
        print('插入错误')
        return False,item