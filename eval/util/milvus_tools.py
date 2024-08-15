from pymilvus import (MilvusClient,connections,FieldSchema, CollectionSchema, DataType, Collection,utility)
import base64, json, requests, glob, tqdm, os,time

class MyMilvus:
    def __init__(self, host, port, username, password,collection_name):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.collection_name = collection_name
        self.connect()

        self.client = MilvusClient(uri="http://{}:{}".format(self.host,self.port),token="{}:{}".format(self.username,self.password))

        # self.nlist = 2048
        self.nlist = 65536
        self.metric_type = "IP"
        # self.metric_type = "L2"
        self.dim = 512

        collections = utility.list_collections()
        if self.collection_name in collections:
            self.collection = Collection(name=self.collection_name)
            print('集合{}已存在，集合中数据量:'.format(self.collection_name),self.get_collection_num())
            # self.drop_collection()

    def drop_collection(self):
        utility.drop_collection(self.collection_name)
        print('集合已删除:',self.collection_name)

    def connect(self):
        connections.connect(
            alias="default",
            uri="http://{}:{}".format(self.host, self.port),
            token="{}:{}".format(self.username, self.password)
        )
        version =utility.get_server_version()
        print(f"connection success,Milvus version: {version}")
        # 连接Milvus服务
        # self.client = MilvusClient(uri="http://{}:{}".format(self.host,self.port),token="{}:{}".format(self.username,self.password))

    def my_create_collection(self):
        schema = MilvusClient.create_schema(auto_id=False,enable_dynamic_field=True,)
        schema.add_field(field_name="_id", datatype=DataType.VARCHAR, max_length=512, is_primary=True)
        schema.add_field(field_name="feature", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="question_text", datatype=DataType.VARCHAR, max_length=65535)
        # schema.add_field(field_name="ori_question", datatype=DataType.VARCHAR, max_length=20480)
        # schema.add_field(field_name="grade", datatype=DataType.INT32)
        # schema.add_field(field_name="subject", datatype=DataType.INT32)
        # schema.add_field(field_name="source_id", datatype=DataType.INT32)
        # schema.add_field(field_name="book_id", datatype=DataType.INT32)
        # schema.add_field(field_name="image_url", datatype=DataType.VARCHAR, max_length=1024)
        # schema.add_field(field_name="question_id", datatype=DataType.VARCHAR, max_length=256)
        self.collection = Collection(name=self.collection_name, schema=schema)
        # self.client.create_collection(collection_name=collection_name, schema=schema)
        # res = self.client.get_load_state(collection_name=collection_name)
        # print(res)

    def get_collection_num(self):
        num = self.collection.num_entities
        return num
    
    def get_question_by_id(self,ids):
        entities = self.client.get(collection_name=self.collection_name, ids=ids)
        questions = []
        for entity in entities:
            question = {}
            question['id'] = entity['_id']
            question['question_text'] = entity['question_text']
            questions.append(question)
        return questions
    
    def id_exists(self, id):
        entities = self.client.get(collection_name=self.collection_name, ids=[id])
        return len(entities) > 0
    
    def index_exists(self):
        index_exists = self.collection.has_index()
        return index_exists
    
    def create_index(self):
        index_params = {"index_type": "IVF_FLAT", "params": {"nlist": self.nlist}, "metric_type": self.metric_type}
        self.collection.create_index(field_name='feature', index_params=index_params)
        print(f"Index created on field '{self.collection_name}' in collection '{self.collection_name}'.")
        self.collection.load()
        print(f"Collection '{self.collection_name}' loaded.")

    def drop_index(self):
        self.collection.drop_index()
        print(f"Index dropped on field '{self.collection_name}' in collection '{self.collection_name}'.")

    def insert_data(self, data):
        mr = self.collection.insert(data, timeout=30)
        # print(f"Inserted data into collection '{self.collection_name}'.")
        return mr
    
    def flush(self):
        self.collection.flush()
        print(f"Collection '{self.collection_name}' flushed.")

    def load(self):
        self.collection.load()
        print(f"Collection '{self.collection_name}' loaded.")

    def query_data(self, query_vectors, top_k=10):
        
        # 搜索参数
        search_params = {
            # "metric_type": "L2",  # 或者 "IP" (Inner Product) 等其他支持的度量类型
            "metric_type": self.metric_type,
            "params": {"nprobe": self.nlist}
        }
        # 执行搜索
        results = self.collection.search(
            data=query_vectors,
            anns_field="feature",  # 向量字段的名称
            param=search_params,
            limit=top_k,
            expr=None  # 可以添加额外的过滤表达式
        )
        return results

