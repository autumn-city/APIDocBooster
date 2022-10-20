import enum
from sqlite3 import apilevel
from subprocess import call
import util.sql as sql
import util.data_preprocess as dp
import json
import requests
import json
import csv
import random

def get_query_name(query):
    return query.split('.')[-2]

def test_for_answer(query):
    answer_id = '67137443'
    html_unit = sql.get_query_body(answer_id)
    codeblock = dp.get_inline_code(html_unit[0]['Body'])
    print(codeblock)

# give the list of inline code of each aswer.
def give_inline_code(query):
    # save dic to json
    
    # get the body of each answer
    body = sql.get_all_body()
    
    print(len(body))

    # with open('data/answer_id.json') as f:
    #     answer_id = json.load(f)

# I decided to give up this idea because it is too slow to get the inline code of each answer. I turn to use the SO api.
def test():

    # initialization
    print("test for single API\n=================")
    query = 'torch.nn.LSTMCell'
    print("query: ", query)
    print("query name: ", get_query_name(query))
    print("\n\n\n\n")

    # exact match the inline code
    # basically, we need to extract the inline code from the body of each answer, 
    # and the compare it with the query name, 
    # if they are the same, then we return the answer id
    give_inline_code(get_query_name(query))

def call_API(query):
    
    # count the page
    i = 1
    # the list of answer
    answer_list = []
    paras = {
        'pagesize': 99,
        'order': 'desc',
        'sort': 'relevance',
        'q': str(query),
        'site': 'stackoverflow',
        # page start from 1
        'page': '1', 
        'answers': '1',
        'key':'tFaahBz1)Kq70INbmCkYrw((',
    }

    print("query: ", paras['q'])

    r = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=paras).json()

    answer_list=r['items']

    while r['has_more'] == True and i < 5:
        i+=1
        paras['page'] = str(i)
        r = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=paras).json()
        answer_list.extend(r['items'])

    # print(answer_list)
    print("total answer: ", len(answer_list))

    # save the answer list to json
    with open('/storage/chengran/APISummarization/data/so_api_relevant_answer_info/answer_meta/'+query+'.json', 'w') as f:
        json.dump(answer_list, f)

# in the form of the list, each element is a dict of the answer info
def read_data(data_path):
    with open(data_path) as f:
        answer_list = json.load(f)
    return answer_list


def get_answers():
    # get the API lists. note that the API names are no replicated
    with open('/storage/chengran/APISummarization/data/torch_related_SO_posts/math_mentioned_api.json') as f:
        api_list = json.load(f)
    # get the answer list for all apis
    for api in api_list:
        api_name = get_query_name(api)
        print("api name: ", api_name)
        call_API(api_name)

def read_answers():
    # read the answer list for all apis as a dic
    answer_dic = {}
    with open('/storage/chengran/APISummarization/data/torch_related_SO_posts/math_mentioned_api.json') as f:
        api_list = json.load(f)
    # get the answer list for all apis
    for api in api_list:
        api_name = get_query_name(api)
        answer_list = read_data('/storage/chengran/APISummarization/data/so_api_relevant_answer_info/answer_meta/'+api_name+'.json')
        answer_dic[api_name] = answer_list
    return answer_dic

def test_answer_extraction(test_answer_dic,query):
    answers = []
    questions = []
    for question in test_answer_dic:
        questions.append(question['question_id'])

    # random.shuffle(questions)
    # questions = questions[:15]

    # question list for norm1d
    # question_list = ['55320883','65398540','56399151','64629522','61193517','48991874','65882526','57114974','47197885']

    # question list for LSTMCell
    answers = ['67137443', '48187516','50050475','72643830','54061903','48147874','62118650','49042283','55233937']

    for num, answer in enumerate(answers):
        # answers = sql.get_answer_from_question_id(question)
        rows = []            

        with open('/storage/chengran/APISummarization/data/so_api_relevant_answer_info/test/question_meta_2/answer_'+str(num)+'.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Block Label','Cluster Label','sentences to be labeled','question title','question link','question description'])
            if sql.get_parientid(answer)!='Null':
                print(sql.get_parientid(answer))
                question = sql.get_parientid(answer)[0]['ParentId']
                title = sql.get_title(question)
                if title:
                    tags = sql.get_tag(question)
                    description = dp.get_description(sql.get_description(question)[0]['body'])
                    answer_item = sql.get_answer_body(answer)
                    sents = dp.split_data(answer_item[0]['Body'])
                    for num, item in enumerate(sents):
                        if item == '[code snippet]' and sents[num-1]=='Paragraph end':
                            sents.pop(num-1)
                            sents.pop(num)
                    for sent in sents:
                        if sent =='[code snippet]' or sent == 'Paragraph end':
                            rows.append(['Null','Null',sent])
                        else:
                            rows.append(['','',sent ,title[0]['Title'],'www.stackoverflow.com/questions/'+str(question),description])
                writer.writerows(rows)





    for question_id in questions:
        for item in sql.get_answer_from_question_id(question_id):
            answers.append(item['id'])
    # randomly select 10 answers
    random.shuffle(answers)
    for answer in answers:
        # id = sql.
        pass


if __name__ == '__main__':
    storage_path = 'data/StackOverflow'

    # call_API('LSTMCell')

    # get the answer list for all apis
    # get_answers()

    # read all the data
    answer_dic = read_answers()
    print("the API names are: ",[(i,len(answer_dic[i])) for i in answer_dic.keys()])
    print('\n\n\n====================\n\n\n')
    
    for i in range(len(answer_dic['LSTMCell'])):
        print(answer_dic['LSTMCell'][i]['question_id'])


    
    # extract the answer content into test labeling data
    test_answer_extraction(answer_dic['LSTMCell'],'LSTMCell')
    
    # extract the setences that mention the API from each answer and store locally
    # for api in answer_dic.keys():
    #     print("api name: ", api)
    #     target_dic = answer_dic[api]
    #     count = 0
    #     sent_API_list = []
    #     for answer in target_dic:
    #         # extract the question id
    #         question_id = answer['question_id']
    #         # extract the answer ids
    #         answer_id = []
    #         for item in sql.get_answer_from_question_id(question_id):
    #             answer_id.append(item['id'])

    #         # extract the API mentioned sentences from each
    #         for id in answer_id:
    #             html_unit = sql.get_query_body(str(id))
    #             content = dp.split_data(html_unit[0]['Body'])
    #             # extract the sentences that mention the API (exact match that is case insensitive)
    #             for sent in content:
    #                 if api.casefold() in sent.casefold():
    #                     # print("answer id: ", id)
    #                     # print("the sentence that mention the API: ", sent)
    #                     count+=1
    #                     sent_API_list.append(sent+'\n')
    #     print("total sentences that mention the API: ", count)
    #     print('\n\n\n====================\n\n\n')
    #     # save the sentences to local
    #     with open('/storage/chengran/APISummarization/data/so_api_relevant_answer_info/sent_info/'+api+'.txt', 'w') as f:
    #         f.writelines(sent_API_list)
            
