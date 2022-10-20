import enum
import os
import nltk
import data.video_api_info.video_analyzer.util.util as util
if __name__ == '__main__':
    # the API name is LSTMCell

    test_path = '/storage/chengran/APISummarization/data/video_api_info/caption_data/test/manua_data'
    total_count = 0
    total_count_lstm = 0
    for file in os.listdir(test_path):
        count = 0
        if file.endswith('.txt'):
            with open(os.path.join(test_path,file), 'r') as f:
                text = f.read().replace('\n',' ')
                sentences = nltk.sent_tokenize(text)
                # print(sentences)
                for num, sent in enumerate(sentences):
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
                print('this video %s has %d sentences' % (file,len(sentences)))
                print('this video %s has %d lstm sentences' % (file,count))
    print('in total there is %d sentences' % total_count)
    print('in total there is %d lstm sentences' % total_count_lstm)

