# In this tool, we don't aim to extract the APIs as much as possible. on the contrary, we aim to extract the "exactly API" without noice data
# 
# Author: Chengran

from email.errors import NoBoundaryInMultipartDefect
import os 
import csv
from pickletools import read_uint1
import re
import sql
import sys
import json
sys.path.append('/storage/chengran/APISummarization/data')

import API_Document.machine_learning.torch_read as source_api

def extract_raw_data(csv_path):
    count = 0
    result = []
    with open(csv_path, ) as f:
        reader = csv.reader(f)
        for post in reader:
            result.append(post[8])
    return result
def extract_raw_answer_data(csv_path):
    count = 0
    result = []
    with open(csv_path, ) as f:
        reader = csv.reader(f)
        for post in reader:
            result.append(post[1])
    return result    
def api_extraction(sent):
    # input: string
    # return: list of APIs
    _return = []
    rule = '(?<!<pre>)<code>([\s\S]*?)<\/code>'
    pattern = re.compile(rule)
    result = pattern.findall(sent)
    if result:
        for api in result:
            if '\n' not in api:
                _return.append(api)
    return _return

def api_classification(apis):
    # given the api candidats, filter out those non-api things, and return the last name of API
    # a real api must satisfy following rules: a.b.c.d; a.b(arg); strings; [type] a(arg); [type] a = b.c(arg); [type] a = b.c
    _return = []
    for api_candidate in apis:

        # relu_1 = '(?<!<pre>)<code>([\s\S]*?)<\/code>'
        if ('.' not in api_candidate) and ('=' not in api_candidate) and (' ' not in api_candidate) and ('(' not in api_candidate) and ('[' not in api_candidate) and ('\\' not in api_candidate) and ('-' not in api_candidate):
            # extract the strings
            _return.append(api_candidate)
        
        if ('.' in api_candidate) and ('=' not in api_candidate) and (' ' not in api_candidate):
            # extract the a.b.c(arg)
            if ('(' in api_candidate):
                _return.append(api_candidate.split("(")[0])
            else:
                # extract a.b.c.d
                # print(api_candidate)
                _return.append(api_candidate)
        # extract the [type] a(arg)
        if ('.' not in api_candidate) and ('=' not in api_candidate) and ('(' in api_candidate):
            # print(api_candidate)
            middle = api_candidate.split("(")[0]
            _return.append(middle.split(' ')[-1])

        # extract the [type] a = b.c(arg)
        if ('=' in api_candidate) and (' ' in api_candidate):
            rule = '(?<=\=).*?(?=\()'
            pattern = re.compile(rule)
            result = pattern.findall(api_candidate)
            # print(api_candidate)
            if result:
                _return.append(result[0].strip())

        # extract the a = b(arg)
    # print(_return)
    __return = []
    for item in _return:
        if '.' in item:
            __return.append(item.split('.')[-1])
        else:
            __return.append(item)
    # print(__return)
    return __return


def initial_match(apis, candidates):
    # count the api frequency in the candidates
    # here we compare the simple API name but not the fully qualified names
    # output: dic [api, frequency]
    _return = {}
    for item in apis:
        count = 0
        for candidate in candidates:
            if item==candidate:
                count+=1
        _return[item]=count
    return _return

def identify_doc(api_short, API_doc_origin_path):
    _return = []
    count = 0
    for doc in os.listdir(API_doc_origin_path):
        if api_short==doc.split('.')[-2]:
            _return.append(doc)
    return _return, os.listdir(API_doc_origin_path)


def frequency_for_math_formulation_SOmention(frequency_largerthan_0, API_doc_origin_path):
    # for each API that has been mentioned in the Stack Overflow posts,
    # identify its original doc and identify whether it have math formulations
    count=0
    num_doc=0
    for api_short in frequency_largerthan_0.keys():
        related_doc = identify_doc(api_short, API_doc_origin_path)
        print('the num of related doc is:%d'%(len(related_doc)))
        tmp_count = 0
        count+=source_api.math_formula_extraction(True, False, related_doc)
        num_doc+=len(related_doc)
    print(related_doc)
    print('the num of math mentioned api docs is :%d'%(count))
    print('the num of all docs are:%d'%(num_doc))

def frequency_for_math_formulation_SOmention_v1(frequency_largerthan_0, API_doc_origin_path):
    # for each API that has been mentioned in the Stack Overflow posts,
    # identify its original doc and identify whether it have math formulations
    count=0
    num_doc=0
    apis_short = []
    result = []
    for api_short in frequency_largerthan_0.keys():
        related_doc,related_doc_full = identify_doc(api_short, API_doc_origin_path)
        print('the num of related doc is:%d'%(len(related_doc)))
        tmp,file=source_api.math_formula_extraction(True, False, related_doc)
        if tmp!=0:
            count+=1
            result.append(file.split('/')[-1])
        num_doc+=len(related_doc)
        apis_short.append(api_short)
    print(related_doc)
    print('the num of math mentioned api docs is :%d'%(count))
    print('the num of all docs are:%d'%(num_doc))
    print('len of unique apis are: '+str(len(list(set(apis_short)))))
    print(result)
    store_as_json(result, '/storage/chengran/APISummarization/data/torch_related_SO_posts/math_mentioned_api.json')
    return 

def store_as_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    question_path = 'QueryResults.csv'
    answer_path = 'AnswerResults.csv'

    result = extract_raw_data(question_path)
    answer_result = extract_raw_answer_data(answer_path)
    candidates = []

    for question in result:
        candidates.extend(api_extraction(question))

    for answer in answer_result:
        candidates.extend(api_extraction(answer))

    # print(candidates)
    candidates = api_classification(candidates)
    apis = source_api.read_API_name()
    # print(apis);exit()

    # short name of apis
    short_apis = [x.split('.')[-1] for x in apis]

    # match function
    frequency = initial_match(short_apis,candidates)
    frequency = dict(sorted(frequency.items(), key = lambda kv:(kv[1], kv[0])))
    frequency_largerthan_0 = {}
    for key in frequency.keys():
        if frequency[key]>0:
            frequency_largerthan_0[key]=frequency[key]
    # print(len(frequency_largerthan_0))
    # print(frequency_largerthan_0)

    # calculate the frequency of math formulation in the SO mentioned APIs
    API_doc_origin_path = '/storage/chengran/APISummarization/data/API_Document/machine_learning/PyTorch.docs/torch/generated/'
    # frequency_for_math_formulation_SOmention(frequency_largerthan_0, API_doc_origin_path)
    frequency_for_math_formulation_SOmention_v1(frequency_largerthan_0, API_doc_origin_path)
