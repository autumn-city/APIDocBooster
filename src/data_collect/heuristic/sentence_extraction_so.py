from subprocess import call
import util.sql as sql
import util.data_preprocess as dp
import json
import requests
import json

def get_query_name(query):
    return query.split('.')[-1]

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
    r = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=paras).json()

    answer_list=r['items']

    while r['has_more']:
        i+=1
        paras['page'] = str(i)
        r = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=paras).json()
        answer_list.extend(r['items'])

    print("total answer: ", len(answer_list))

if __name__ == '__main__':
    storage_path = 'data/StackOverflow'
    # test()
    call_API('LSTMCell')
