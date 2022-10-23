import enum
import os
import nltk
import util.util as util
from pathlib import Path
import csv
import sys
import copy

sys.path.append("..")
from search_video_given_api.YouTube_Search_API.youtube_search import search_video_title
def doc_statistic(doc):
    total_count = 0
    total_count_lstm = 0
    for num, sent in enumerate(doc):
        # print(sent)
        if 'lstmcell' in sent.lower() or 'input_size' in sent.lower() or 'hidden_size' in sent.lower() or 'bias' in sent.lower():
            # print(sentences[num-1]+'\n')
            # print(sent+'\n')
            # print(sentences[num+1]+'\n')
            count+=1
            total_count_lstm+=1
            print(sent+'\n')
            # print(sentences[num-1]+'\n')
            # print(sentences[num+1]+'\n')
            # exit()
        total_count+=1
            # print('=====================')
    print('this video %s has %d sentences' % (file,len(doc)))
    print('this video %s has %d lstm sentences' % (file,count))

def extract_relevant_sentences(doc_rank, doc, top_k_sent):
    #given the bm25 score, extract the sentences which score is higher than 0 and sentences that are context to high score sentences. 
    high_score_sent = {}
    candidate_sent = {}
    return_list = {}
    for num, sent in enumerate(doc):
        if doc_rank[num]>0:
            high_score_sent[num] = doc_rank[num]
    sorted_sent = sorted(high_score_sent.items(), key=lambda x: x[1], reverse=True)[:top_k_sent]

    
    for order, score in sorted_sent[:50]:

        # candidate_sent.append(num)
        if order-1<len(doc):
            # print('the before sentence is : ',doc[num-1])
            candidate_sent[order-1] = doc[order-1]
        # print('the current sentence is : ',sent)
        candidate_sent[order] = doc[order]
        if order+1<len(doc):
            # print('the after sentence is : ',doc[num+1])
            candidate_sent[order+1] = doc[order+1]
    for i in sorted (candidate_sent) : 
        # print ((i, candidate_sent[i]), end ="\n") 
        return_list[i] = candidate_sent[i]
  
    return return_list

def pick_top_video(test_path, topk):
    # to reduce the number of the videos, we pick the top 5 videos, in each video the captions don't contain too long sentences.
    candidate = []
    for file in os.listdir(test_path):
        if file.endswith('.txt'):
            with open(os.path.join(test_path,file), 'r') as f:
                num_sent = 0
                text = f.read().replace('\n',' ')
                sentences = nltk.sent_tokenize(text)
                for item in sentences:
                    num_sent+=len(item)
                if (num_sent/len(sentences))<=500:
                    candidate.append(os.path.join(test_path,file))
    print(candidate.sort(key=os.path.getctime))
    return candidate[:topk]
                    

def return_key(val,dic):
    for key, value in dic.items():
        if value==val:
            return key
    return('Key Not Found')

def sentence_to_label_csv(useful_sent,video_titles):
    with open('/storage/chengran/APISummarization/data/video_api_info/pilot_study/lstmcell.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Block Label','Cluster Label','sentences to be labeled','video title','video link','video description'])
        rows = []
        for order, sent in [(k, v) for k, v in useful_sent.items()]:
            videolink = ''
            last_order = 0
            for num, video_info in enumerate(video_titles.values()):
                if video_info>order:
                    # print('the video order is ', order)
                    # print('the video max number is ', video_info)
                    videolink = return_key(video_info,video_titles)
                    # print('the video link is ', videolink)
                    # print(videolink.split('/')[-1].split('.')[0])
                    if num>0:
                        last_order = list(video_titles.values())[num-1]
                    break
            video_link = 'https://www.youtube.com/watch?v='+videolink.split('/')[-1].split('.')[0]
            # print('the video link is ', video_link)
            video_title = search_video_title(videolink.split('/')[-1].split('.')[0])['title']
            # print('the video title is ', video_title)
            video_description = search_video_title(videolink.split('/')[-1].split('.')[0])['description']
            rows.append(['','',str(order-last_order)+' : '+sent,video_title,video_link,video_description])
        rows_1 = copy.deepcopy(rows)

        # store data into pickel file
        # util.store_data(rows_1, '/storage/chengran/APISummarization/data/video_api_info/pilot_study/lstmcell.pkl')

        num_video = 0
        for order, row in enumerate(rows):
            if order>0:
                # print(row[4])
                # print(rows[order-1][4])
                if row[4]!=rows[order-1][4]:
                    rows_1.insert(order+num_video, ['','','new video'])
                    num_video+=1
                    print('the order of the sentence that is new video', order)
                    print('the current order is ', order+num_video) 
                    print('the num_video is ', num_video)
                    print('\n')                 
        writer.writerows(rows_1)
    exit()
            



            # question = sql.get_parientid(answer)[0]['ParentId']
            # title = sql.get_title(question)
            # if title:
            #     tags = sql.get_tag(question)
            #     description = dp.get_description(sql.get_description(question)[0]['body'])
            #     answer_item = sql.get_answer_body(answer)
            #     sents = dp.split_data(answer_item[0]['Body'])
            #     for num, item in enumerate(sents):
            #         if item == '[code snippet]' and sents[num-1]=='Paragraph end':
            #             sents.pop(num-1)
            #             sents.pop(num)
            #     for sent in sents:
            #         if sent =='[code snippet]' or sent == 'Paragraph end':
            #             rows.append(['Null','Null',sent])
            #         else:
            #             rows.append(['','',sent ,title[0]['Title'],'www.stackoverflow.com/questions/'+str(question),description])
            # writer.writerows(rows)



if __name__ == '__main__':
    # the API name is LSTMCell

    test_path = '/storage/chengran/APISummarization/data/video_api_info/caption_data/test/manua_data'
    # pick the top-k videos to minimize the number of videos
    top_k_video = 10
    # pick the top-k sentences to minimize the number of sentences in terms of bm25 score
    top_k_sent = 50
    candidates = pick_top_video(test_path,10)
    sent_source = []
    # used to extract the video title and description
    video_titles = {}
    sent_count = 0
    for file in candidates:
        count = 0
        if file.endswith('.txt'):
            with open(os.path.join(test_path,file), 'r') as f:
                text = f.read().replace('\n',' ')
                sentences = nltk.sent_tokenize(text)
                sent_source.extend(sentences)
                sent_count+=len(sentences)
                video_titles[file]=sent_count
    # print(len(sent_source));exit()
    query = 'lstm cell input_size hidden_size bias'
    doc_rank = util.bm25_search(sent_source, query)
    useful_sent =extract_relevant_sentences(doc_rank, sent_source, top_k_sent)
    sentence_to_label_csv(useful_sent,video_titles)
    # print('1')


