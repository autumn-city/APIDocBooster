import enum
import os
import nltk
import util.util as util
from pathlib import Path

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
    candidate_sent = {}
    for num, sent in enumerate(doc):
        if doc_rank[num] > 0:
            # candidate_sent.append(num)
            if num-1<len(doc):
                # print('the before sentence is : ',doc[num-1])
                candidate_sent[num-1] = doc_rank[num-1]
            # print('the current sentence is : ',sent)
            candidate_sent[num] = doc_rank[num]
            if num+1<len(doc):
                # print('the after sentence is : ',doc[num+1])
                candidate_sent[num+1] = doc_rank[num+1]
    print(candidate_sent);exit()
        
    # print('=====================\n')
    return len(candidate_sent)

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
                    

if __name__ == '__main__':
    # the API name is LSTMCell

    test_path = '/storage/chengran/APISummarization/data/video_api_info/caption_data/test/manua_data'
    total_count = 0
    # pick the top-k videos to minimize the number of videos
    top_k_video = 10
    # pick the top-k sentences to minimize the number of sentences in terms of bm25 score
    top_k_sent = 10
    candidates = pick_top_video(test_path,10)

    for file in candidates:
        count = 0
        if file.endswith('.txt'):
            with open(os.path.join(test_path,file), 'r') as f:
                text = f.read().replace('\n',' ')
                sentences = nltk.sent_tokenize(text)
                query = 'lstm cell input_size hidden_size bias'
                doc_rank = util.bm25_search(sentences, query)
                total_count+=extract_relevant_sentences(doc_rank, sentences, top_k_sent)
    print(total_count)


