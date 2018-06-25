from elasticsearch import helpers, Elasticsearch
import csv
import requests

res = requests.get("http://ec2-18-216-228-96.us-east-2.compute.amazonaws.com:9200")
print(res.content)

es = Elasticsearch([{"host": "ec2-18-216-228-96.us-east-2.compute.amazonaws.com",
                     "port": 9200}])

if es.indices.exists(index="tweets-data"):
    pass
else:
    es.indices.create(index="tweets-data")

with open("tweets_with_sentiments.csv", encoding="utf8") as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index="tweets-data", doc_type="tweets")


# es.get(index='tweets-data', doc_type='tweets')
es.search(index="tweets-data", body={"query": {"match": {'sentiment':'neutral'}}})
