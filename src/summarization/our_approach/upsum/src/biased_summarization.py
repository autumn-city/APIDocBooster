import json
from os import listdir
from os.path import isfile, join
import os
import nltk
import numpy as np
from biases import democratic_bias, republican_bias
import pickle
import csv
# from rouge import Rouge
# from pyrouge import Rouge155

from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('bert-base-nli-mean-tokens')
# sbert = SentenceTransformer('bert-base-nli-mean-tokens', device='cuda:1') uncomment for the gpt-2 generation
# rouge = Rouge155()

def get_sentence_simcsse_similarity_update(input):
    embeddings = []

    from simcse import SimCSE
    model = SimCSE("../model/my-sup-simcse-bert-base-uncased/")
    for sent in input:
        embedding = model.encode(sent)
        embeddings.append(embedding)
    return embeddings

def vcosine(u, v):
    return abs(1 - distance.cdist(u, v, 'cosine'))


def cosine(u, v):
    return abs(1 - distance.cosine(u, v))


def rescale(a):
    maximum = np.max(a)
    minimum = np.min(a)
    return (a - minimum) / (maximum - minimum)

def calculate_bias_weights(texts_embeddings, api_doc_embedding):
    return vcosine(texts_embeddings, api_doc_embedding)


def update_textrank(texts_embeddings, bias_embedding, api_doc_embedding, specification, damping_factor=0.7, similarity_threshold=0.9, biased=True):

    print('the query embeeding shape is : ', len(bias_embedding))
    print('the api doc embeeding shape is : ', len(api_doc_embedding))

    matrix = vcosine(texts_embeddings, texts_embeddings)
    np.fill_diagonal(matrix, 0)
    matrix[matrix < similarity_threshold] = 0

    matrix = normalize(matrix)

    query_bias_weights = calculate_bias_weights(texts_embeddings, bias_embedding)
    query_bias_weights = rescale(query_bias_weights)

    apidoc_bias_weights = calculate_bias_weights(texts_embeddings, api_doc_embedding)
    apidoc_bias_weights = rescale(apidoc_bias_weights)

    # print('the bias weight shape is: ', bias_weights.shape)
    # print('the summarization matrix shape is: ', matrix.shape)

    if biased:
        iterations = 100
        # bias_weights = vcosine(bias_embedding, texts_embeddings)
        # bias_weights = rescale(bias_weights)
        # scaled_matrix = damping_factor * matrix + (1 - damping_factor)
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) * apidoc_bias_weights


    scaled_matrix = normalize(scaled_matrix)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    for i in range(iterations):
        ranks = scaled_matrix.T.dot(ranks)

    return ranks

def update_textrank_new(texts_embeddings, bias_embedding, api_doc_embedding, specification, damping_factor=0.85, similarity_threshold=0.8, biased=True):

    print(len(texts_embeddings))

    matrix = vcosine(texts_embeddings, texts_embeddings)
    np.fill_diagonal(matrix, 0)
    matrix[matrix < similarity_threshold] = 0

    matrix = normalize(matrix)

    bias_weights = calculate_bias_weights(texts_embeddings, api_doc_embedding)
    print(bias_weights.shape)

    bias_weights = rescale(bias_weights)
    bias_weights = np.mean(bias_weights,axis=1)
    bias_weights = np.reshape(bias_weights, (-1, 1))
    print(matrix.shape)
    print(bias_weights.shape)

    if biased:
        if specification=='function':
            iterations = 100
            # bias_weights = vcosine(bias_embedding, texts_embeddings)
            # bias_weights = rescale(bias_weights)
            # scaled_matrix = damping_factor * matrix + (1 - damping_factor)
            scaled_matrix = damping_factor * matrix + (1 - damping_factor) * (1-bias_weights)

        if specification!='function':
            iterations = 100
            # bias_weights = vcosine(bias_embedding, texts_embeddings)
            # bias_weights = rescale(bias_weights)
            # scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias_weights
            scaled_matrix = damping_factor * matrix + (1 - damping_factor)

    scaled_matrix = normalize(scaled_matrix)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    for i in range(iterations):
        ranks = scaled_matrix.T.dot(ranks)

    return ranks


def biased_textrank(texts_embeddings, bias_embedding, api_doc_embedding, damping_factor=0.7, similarity_threshold=0.7, biased=True):

    print(len(texts_embeddings))

    matrix = vcosine(texts_embeddings, texts_embeddings)
    np.fill_diagonal(matrix, 0)
    matrix[matrix < similarity_threshold] = 0

    matrix = normalize(matrix)

    if biased:
        iterations = 100
        bias_weights = vcosine(bias_embedding, texts_embeddings)
        bias_weights = rescale(bias_weights)
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias_weights
    else:
        iterations = 100
        damping_factor = 0.9
        scaled_matrix = damping_factor * matrix + (1 - damping_factor)

    scaled_matrix = normalize(scaled_matrix)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    for i in range(iterations):
        ranks = scaled_matrix.T.dot(ranks)

    return ranks


def biased_textrank_ablation(texts_embeddings, bias_embedding, damping_factors=[0.8, 0.85, 0.9],
                             similarity_thresholds=[0.7, 0.75, 0.8, 0.85, 0.9]):
    main_matrix = vcosine(texts_embeddings, texts_embeddings)
    np.fill_diagonal(main_matrix, 0)

    bias_weights = vcosine(bias_embedding, texts_embeddings)
    bias_weights = rescale(bias_weights)

    ranks = {}
    for similarity_threshold in similarity_thresholds:
        if similarity_threshold not in ranks:
            ranks[similarity_threshold] = {}
        matrix = main_matrix.copy()
        # removing edges that don't pass the similarity threshold
        matrix[matrix < similarity_threshold] = 0
        # matrix = normalize(matrix)
        for damping_factor in damping_factors:
            scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias_weights
            scaled_matrix = normalize(scaled_matrix)

            print('Calculating ranks for sim thresh {} damping {}'.format(similarity_threshold, damping_factor))
            _ranks = np.ones((len(matrix), 1)) / len(matrix)
            iterations = 80
            for i in range(iterations):
                _ranks = scaled_matrix.T.dot(_ranks)

            ranks[similarity_threshold][damping_factor] = _ranks

    return ranks


def normalize(matrix):
    for row in matrix:
        row_sum = np.sum(row)
        if row_sum != 0:
            row /= row_sum
    return matrix


def get_sbert_embedding(text):
    if isinstance(text, list) or isinstance(text, tuple):
        return sbert.encode(text)
    else:
        return sbert.encode([text])


def get_sentences(text):
    paragraphs = text.split('\n\n')
    sentences = []
    for paragraph in paragraphs:
        sentences += nltk.sent_tokenize(paragraph)
    sentences = [s for s in sentences if s and not s.isspace()]
    return sentences


def load_text_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data


def get_filenames_in_directory(path):
    return [filename for filename in listdir(path) if isfile(join(path, filename))]


def select_top_k_texts_preserving_order(texts, ranking, k):
    texts_sorted = sorted(zip(texts, ranking), key=lambda item: item[1], reverse=True)
    top_texts = texts_sorted[:k]
    top_texts = [t[0] for t in top_texts]
    result = []
    for text in texts:
        if text in top_texts:
            result.append(text)
    return result

def get_ground_truth_sum_length(file):
    return len(open(file,'r').readlines()) 


def load_ground_truth_data(democrat_path, republican_path, transcript_path):
    democrat_gold_standards = [{'filename': filename, 'content': load_text_file(democrat_path + filename)} for filename
                               in get_filenames_in_directory(democrat_path)]
    republican_gold_standards = [{'filename': filename, 'content': load_text_file(republican_path + filename)} for
                                 filename in get_filenames_in_directory(republican_path)]
    transcripts = [{'filename': filename, 'content': load_text_file(transcript_path + filename)}
                   for filename in get_filenames_in_directory(transcript_path)]
    return democrat_gold_standards, republican_gold_standards, transcripts


def get_data_paths():
    data_path = '../data/us-presidential-debates/'
    democrat_path = data_path + 'democrat/'
    republican_path = data_path + 'republican/'
    transcript_path = data_path + 'transcripts/'
    return democrat_path, republican_path, transcript_path


def get_bias_embeddings():
    democratic_bias_embedding = get_sbert_embedding(democratic_bias)
    republican_bias_embedding = get_sbert_embedding(republican_bias)
    return democratic_bias_embedding, republican_bias_embedding


def ablation_study():
    democratic_bias_embedding, republican_bias_embedding = get_bias_embeddings()

    with open('sentence_and_embeddings_checkpoints.json') as f:
        sentences_and_embeddings = json.load(f)
    # sentences_and_embeddings = sentences_and_embeddings[:1]

    for item in sentences_and_embeddings:
        item['embeddings'] = np.array(item['embeddings'])

    damping_factors = [0.8, 0.85, 0.9]
    similarity_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    democratic_summaries = [{'filename': item['filename']} for item in sentences_and_embeddings]
    republican_summaries = [{'filename': item['filename']} for item in sentences_and_embeddings]

    for i, item in enumerate(sentences_and_embeddings):
        sentences = item['sentences']
        embeddings = item['embeddings']
        democratic_summaries[i]['content'] = {}
        republican_summaries[i]['content'] = {}
        democratic_ranks = biased_textrank_ablation(embeddings, democratic_bias_embedding,
                                                    damping_factors=damping_factors,
                                                    similarity_thresholds=similarity_thresholds)
        republican_ranks = biased_textrank_ablation(embeddings, republican_bias_embedding,
                                                    damping_factors=damping_factors,
                                                    similarity_thresholds=similarity_thresholds)
        for similarity_threshold in similarity_thresholds:
            democratic_summaries[i]['content'][similarity_threshold] = {}
            republican_summaries[i]['content'][similarity_threshold] = {}
            for damping_factor in damping_factors:
                _democratic_ranks = democratic_ranks[similarity_threshold][damping_factor]
                democratic_summaries[i]['content'][similarity_threshold][damping_factor] = ' '.join(
                    select_top_k_texts_preserving_order(sentences, _democratic_ranks, 30))
                _republican_ranks = republican_ranks[similarity_threshold][damping_factor]
                republican_summaries[i]['content'][similarity_threshold][damping_factor] = ' '.join(
                    select_top_k_texts_preserving_order(sentences, _republican_ranks, 30))

    democrat_path, republican_path, transcript_path = get_data_paths()
    democrat_gold_standards, republican_gold_standards, transcripts = load_ground_truth_data(democrat_path,
                                                                                             republican_path,
                                                                                             transcript_path)

    # saving results
    # with open('focused_summarization_ablation.json', 'w') as f:
    #     all_results = {
    #         'democrat': democratic_summaries,
    #         'republican': republican_summaries
    #     }
    #     f.write(json.dumps(all_results))

    rouge_results = {}
    for similarity_threshold in similarity_thresholds:
        rouge_results[similarity_threshold] = {}
        for damping_factor in damping_factors:
            rouge_results[similarity_threshold][damping_factor] = {}
            dem_summaries = [
                {'filename': item['filename'], 'content': item['content'][similarity_threshold][damping_factor]} for
                item in democratic_summaries]
            rep_summaries = [
                {'filename': item['filename'], 'content': item['content'][similarity_threshold][damping_factor]} for
                item in republican_summaries]
            democrat_rouge_scores = calculate_rouge_score(democrat_gold_standards, dem_summaries)
            print('Similarity Threshold={}, Damping Factor={}'.format(similarity_threshold, damping_factor))
            print('Democrat Results:')
            print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(democrat_rouge_scores['rouge-1']),
                                                                 np.mean(democrat_rouge_scores['rouge-2']),
                                                                 np.mean(democrat_rouge_scores['rouge-l'])))

            rouge_results[similarity_threshold][damping_factor]['democrat'] = [
                np.mean(democrat_rouge_scores['rouge-1']), np.mean(democrat_rouge_scores['rouge-2']),
                np.mean(democrat_rouge_scores['rouge-l'])]

            republican_rouge_scores = calculate_rouge_score(republican_gold_standards, rep_summaries)
            print('Republican Results:')
            print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(republican_rouge_scores['rouge-1']),
                                                                 np.mean(republican_rouge_scores['rouge-2']),
                                                                 np.mean(republican_rouge_scores['rouge-l'])))
            print('############################')

            rouge_results[similarity_threshold][damping_factor]['republican'] = [
                np.mean(republican_rouge_scores['rouge-1']), np.mean(republican_rouge_scores['rouge-2']),
                np.mean(republican_rouge_scores['rouge-l'])]

    with open('focused_summarization_rouge.json', 'w') as f:
        f.write(json.dumps(rouge_results))


def calculate_rouge_score(gold_standards, summaries):
    democrat_rouge_scores = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': []
    }
    for gold_standard in gold_standards:
        if len(gold_standard['content']) == 0:
            continue
        for summary in summaries:
            if summary['filename'] == gold_standard['filename']:
                rouge_scores = rouge.get_scores(summary['content'], gold_standard['content'])
                democrat_rouge_scores['rouge-1'].append(rouge_scores[0]['rouge-1']['f'])
                democrat_rouge_scores['rouge-2'].append(rouge_scores[0]['rouge-2']['f'])
                democrat_rouge_scores['rouge-l'].append(rouge_scores[0]['rouge-l']['f'])
    return democrat_rouge_scores

def biased_textrank_run():
    documents_dir = '/workspace/data/labeling/collection'

    specifications = ['function','parameter','others']

    for index, file in enumerate(os.listdir(documents_dir)): 

        try:

            with open(os.path.join(documents_dir,file,'doc.txt'),'r') as f:
                content = [line for line in f.read().splitlines()]
                api_doc_embedding = get_sentence_simcsse_similarity_update(content)
                api_doc_embedding = [t.numpy() for t in api_doc_embedding]
        except:
            pass


        for index, specification in enumerate(specifications):
            resource = []
            # query = file.split('_')[1][:-4]
            query = file

            sum_length = get_ground_truth_sum_length(os.path.join(documents_dir,file,specification+'.txt'))

            with open(os.path.join(documents_dir,file,specification+'.csv'),'r') as f:
                import csv
                reader = csv.reader(f)
                for item in reader:           
                    resource.append(item[0]) 

            transcript_sentence_embeddings = get_sentence_simcsse_similarity_update(resource)
            transcript_sentence_embeddings = [t.numpy() for t in transcript_sentence_embeddings]

            democratic_bias_embedding = get_sentence_simcsse_similarity_update(query)
            democratic_bias_embedding = [t.numpy() for t in democratic_bias_embedding]
            # print(transcript_sentence_embeddings);exit()
            try:
                # democratic_ranks = biased_textrank(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, biased = False)
                democratic_ranks = update_textrank(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, specification, biased = True)

                
            except:
                print(file)
                # print(query)
                # print('====\n')
                # print(resource)
                # exit()
            # print(democratic_ranks);exit()
            democrat_summary = select_top_k_texts_preserving_order(resource, democratic_ranks, sum_length)
            # print(len(democrat_summary))
            # with open('/workspace/APISummarization/src/summarization/biased_textrank/result/function/'+file+'_function_nobias.txt','w') as f:
            if not os.path.exists('/workspace/src/summarization/our_approach/biased_textrank/result/'+file):
                os.mkdir('/workspace/src/summarization/our_approach/biased_textrank/result/'+file)
            with open('/workspace/src/summarization/our_approach/biased_textrank/result/'+file+'/'+str(specification)+'_nobias.txt','w') as f:
                for sent in democrat_summary:
                    f.write(sent+'\n')
    os.remove('/workspace/src/summarization/our_approach/biased_textrank/result/onPause/parameter_nobias.txt')

def biased_textrank_run_copy(damping_factor, similarity_threshold):
    documents_dir = '/workspace/data/labeling/collection'

    specifications = ['function','parameter','others']

    for index, file in enumerate(os.listdir(documents_dir)): 

        try:
            # # treat each sentence as a vector
            # with open(os.path.join(documents_dir,file,'doc.txt'),'r') as f:
            #     content = [line for line in f.read().splitlines()]
            #     api_doc_embedding = get_sentence_simcsse_similarity_update(content)
            #     api_doc_embedding = [t.numpy() for t in api_doc_embedding]

            # # treat all sentences as a vector
            contents = ''
            with open(os.path.join(documents_dir,file,'doc.txt'),'r') as f:
                content = [line for line in f.read().splitlines()]
                for line in f.read().splitlines():
                    contents+=line
                print(contents)
                api_doc_embedding = get_sentence_simcsse_similarity_update([contents])
                api_doc_embedding = [t.numpy() for t in api_doc_embedding]          
        except:
            pass


        for index, specification in enumerate(specifications):
            resource = []
            # query = file.split('_')[1][:-4]
            query = file


            sum_length = get_ground_truth_sum_length(os.path.join(documents_dir,file,specification+'.txt'))

            with open(os.path.join(documents_dir,file,specification+'.csv'),'r') as f:
                import csv
                reader = csv.reader(f)
                for item in reader:           
                    resource.append(item[0]) 

            transcript_sentence_embeddings = get_sentence_simcsse_similarity_update(resource)
            transcript_sentence_embeddings = [t.numpy() for t in transcript_sentence_embeddings]

            print('the length of the query is ', len(query))
            print([query])

            democratic_bias_embedding = get_sentence_simcsse_similarity_update([query])
            democratic_bias_embedding = [t.numpy() for t in democratic_bias_embedding]
            try:
                democratic_ranks = update_textrank(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, 
                specification, damping_factor, similarity_threshold, biased = True)
            except:
                print('no summarization made for this specification')

            democrat_summary = select_top_k_texts_preserving_order(resource, democratic_ranks, sum_length)
            # print(len(democrat_summary))
            # with open('/workspace/APISummarization/src/summarization/biased_textrank/result/function/'+file+'_function_nobias.txt','w') as f:
            path = '/workspace/src/summarization/our_approach/ablation_result/'+str(damping_factor)+'_'+str(similarity_threshold) 
            if not os.path.exists(path):
                os.mkdir(path)
            if not os.path.exists(os.path.join(path,file)):
                os.mkdir(os.path.join(path,file))
            with open(path+'/'+file+'/'+str(specification)+'_nobias.txt','w') as f:
                for sent in democrat_summary:
                    f.write(sent+'\n')
    os.remove(path+'/onPause/parameter_nobias.txt')



def biased_textrank_end2end():
    data_dir = '/workspace/src/classification/result'
    documents_dir = '/workspace/data/labeling/collection'

    specifications = ['function','parameter','others']

    for index, file in enumerate(os.listdir(data_dir)): 

        # try:

        with open(os.path.join(documents_dir,file,'doc.txt'),'r') as f:
            print(file)
            content = [line for line in f.read().splitlines()]
            api_doc_embedding = get_sentence_simcsse_similarity_update(content)
            api_doc_embedding = [t.numpy() for t in api_doc_embedding]
        # except:
            # print('error')
        
        # exit()

        for index, specification in enumerate(specifications):
            resource = []
            # query = file.split('_')[1][:-4]
            query = file

            sum_length = get_ground_truth_sum_length(os.path.join(documents_dir,file,specification+'.txt'))

            with open(os.path.join(data_dir,file,specification+'.txt'),'r') as f:
                resource=f.readlines()

            transcript_sentence_embeddings = get_sentence_simcsse_similarity_update(resource)
            transcript_sentence_embeddings = [t.numpy() for t in transcript_sentence_embeddings]

            democratic_bias_embedding = get_sentence_simcsse_similarity_update(query)
            democratic_bias_embedding = [t.numpy() for t in democratic_bias_embedding]
            # print(transcript_sentence_embeddings);exit()
            try:
                # democratic_ranks = biased_textrank(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, biased = False)
                democratic_ranks = update_textrank_new(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, specification, biased = True)

                
            except:
                print(file)
                # print(query)
                # print('====\n')
                # print(resource)
                # exit()
            # print(democratic_ranks);exit()
            democrat_summary = select_top_k_texts_preserving_order(resource, democratic_ranks, sum_length)
            # print(len(democrat_summary))
            # with open('/workspace/APISummarization/src/summarization/biased_textrank/result/function/'+file+'_function_nobias.txt','w') as f:
            if not os.path.exists('/workspace/src/summarization/result/'+file):
                os.mkdir('/workspace/src/summarization/result/'+file)
            with open('/workspace/src/summarization/result/'+file+'/'+str(specification)+'_nobias.txt','w') as f:
                for sent in democrat_summary:
                    f.write(sent+'\n')

def biased_textrank_baseline():
    data_dir = '/workspace/src/classification/baseline_result'
    documents_dir = '/workspace/data/labeling/collection'

    specifications = ['function','parameter','others']

    for index, file in enumerate(os.listdir(data_dir)): 

        # try:

        with open(os.path.join(documents_dir,file,'doc.txt'),'r') as f:
            print(file)
            content = [line for line in f.read().splitlines()]
            api_doc_embedding = get_sentence_simcsse_similarity_update(content)
            api_doc_embedding = [t.numpy() for t in api_doc_embedding]
        # except:
            # print('error')
        
        # exit()

        for index, specification in enumerate(specifications):
            resource = []
            # query = file.split('_')[1][:-4]
            query = file

            sum_length = get_ground_truth_sum_length(os.path.join(documents_dir,file,specification+'.txt'))

            with open(os.path.join(data_dir,file,specification+'.txt'),'r') as f:
                resource=f.readlines()

            transcript_sentence_embeddings = get_sentence_simcsse_similarity_update(resource)
            transcript_sentence_embeddings = [t.numpy() for t in transcript_sentence_embeddings]

            democratic_bias_embedding = get_sentence_simcsse_similarity_update(query)
            democratic_bias_embedding = [t.numpy() for t in democratic_bias_embedding]
            # print(transcript_sentence_embeddings);exit()
            try:
                # democratic_ranks = biased_textrank(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, biased = False)
                democratic_ranks = update_textrank_new(transcript_sentence_embeddings, democratic_bias_embedding, api_doc_embedding, specification, biased = True)

                
            except:
                print(file)
                # print(query)
                # print('====\n')
                # print(resource)
                # exit()
            # print(democratic_ranks);exit()
            democrat_summary = select_top_k_texts_preserving_order(resource, democratic_ranks, sum_length)
            # print(len(democrat_summary))
            # with open('/workspace/APISummarization/src/summarization/biased_textrank/result/function/'+file+'_function_nobias.txt','w') as f:
            if not os.path.exists('/workspace/src/summarization/baseline_result/'+file):
                os.mkdir('/workspace/src/summarization/baseline_result/'+file)
            with open('/workspace/src/summarization/baseline_result/'+file+'/'+str(specification)+'_nobias.txt','w') as f:
                for sent in democrat_summary:
                    f.write(sent+'\n')


def textrank():
    import spacy
    nlp = spacy.load("en_core_web_sm")


if __name__ == '__main__':
    # main()

    # biased_textrank_run()
    damping_factors = [0.6,0.7,0.8,0.9]
    similarity_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for damping_factor in damping_factors:
        for similarity_threshold in similarity_thresholds:
            biased_textrank_run_copy(damping_factor, similarity_threshold)

    # biased_textrank_end2end()
    # biased_textrank_baseline()
