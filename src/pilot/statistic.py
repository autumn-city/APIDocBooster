import os
import json

import pymysql

mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="123456",
    database="SO"
)

cursor = mydb.cursor()


def count_given_so_post_id(post_id):

    count_length = 0

    # get answer id
    sql_get_anwer_ids = 'SELECT Id FROM posts WHERE PostTypeId = 2 AND ParentId = '+str(post_id)+';'
    cursor.execute(sql_get_anwer_ids)
    result = cursor.fetchall()
    # print(result)
    tmps = []
    for row in result:
        tmps.append(str(row[0]))

    # 3. given each answer id, get the length of string in the body
    for id in tmps:
        sql_get_answer_length ='SELECT LENGTH(Body) AS AnswerBodyLength FROM posts WHERE Id = '+str(id)+' AND PostTypeId = 2;'
        cursor.execute(sql_get_answer_length)
        result = cursor.fetchall()
        count_length += result[0][0]

    sql_get_post_length ='SELECT LENGTH(Body) AS AnswerBodyLength FROM posts WHERE Id = '+str(post_id)+' AND PostTypeId = 1'
    cursor.execute(sql_get_post_length)
    result = cursor.fetchall()
    try:
        count_length += result[0][0]
    except:
        pass
    return count_length

def count_relevant_so_posts():

    api_num = 0

    answers_num = 0

    string_num = 0

    so_post_path = '/storage/chengran/APISUM_replication/src/pilot/SO_search_data'
    for api in os.listdir(so_post_path):
        # read json file
        with open(os.path.join(so_post_path, api), 'r') as f:
            data = json.load(f)
            for item in data:
                string_num+=count_given_so_post_id(item['question_id'])

            # api_num += 1
            answers_num += len(data)
    
    # print('api_num: ', api_num)
    # print('answers_num: ', answers_num)
    # print('average answers per api: ', answers_num/api_num)
    print('string_num: ', string_num)
    print('average string length: ', string_num/answers_num)

def count_relevant_yt_posts():
    import pickle
    yt_path = '/storage/chengran/APISUM_replication/src/pilot/YouTube_search_data'

    api_num = 0
    video_num = 0

    for api in os.listdir(yt_path):
        with open(os.path.join(yt_path, api), 'rb') as f:
            try:
                data = pickle.load(f)
            except:
                print('error: ', api)
                continue
            api_num += 1
            video_num += len(data) 
    
    print('api_num: ', api_num)
    print('video_num: ', video_num)
    print('average videos per api: ', video_num/api_num)

def count_yt_posts_length():
    generation_caption_path = '/storage/chengran/APISUM_replication/src/pilot/YouTube_caption_data/test/generated_data'
    manual_caption_path = '/storage/chengran/APISUM_replication/src/pilot/YouTube_caption_data/test/manua_data'
    caption_num = 0
    caption_length = 0
    for api in os.listdir(generation_caption_path):
        # read txt file
        with open(os.path.join(generation_caption_path, api), 'r') as f:
            data = f.read()
            caption_num += 1
            caption_length += len(data)
    for api in os.listdir(manual_caption_path):
        for doc in os.listdir(os.path.join(manual_caption_path, api)):
            # read txt file
            with open(os.path.join(manual_caption_path, api, doc), 'r') as f:
                data = f.read()
                caption_num += 1
                caption_length += len(data)
    
    print('caption_num: ', caption_num)
    print('caption_length: ', caption_length)
    print('average caption length: ', caption_length/caption_num)





# main function
def main():
    # count_relevant_so_posts()
    # count_relevant_yt_posts()
    count_yt_posts_length()

if __name__ == '__main__':
    main()