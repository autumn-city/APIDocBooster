import openai
import json
import time
from sklearn.metrics import f1_score
import argparse
import logging
import pandas as pd
import numpy as np
import sys
import time
import os
import csv
from nltk import tokenize
import tiktoken
import pprint
import re

def replace_whitespace_with_space(text):
    # Replace all whitespace characters with ' '
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text

def accuracy(preds, labels):
    correct = sum(int(pred == label) for pred, label in zip(preds, labels))
    total = len(labels)
    accuracy = correct / total
    return accuracy

def precision(preds, labels):
    true_positives = sum(int(pred == 1 and label == 1) for pred, label in zip(preds, labels))
    false_positives = sum(int(pred == 1 and label == 0) for pred, label in zip(preds, labels))
    precision = true_positives / (true_positives + false_positives)
    return precision

def recall(preds, labels):
    true_positives = sum(int(pred == 1 and label == 1) for pred, label in zip(preds, labels))
    false_negatives = sum(int(pred == 0 and label == 1) for pred, label in zip(preds, labels))
    recall = true_positives / (true_positives + false_negatives)
    return recall

def evaluator(preds, labels):

    pos_correct_count, neg_correct_count = 0,0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            if labels[i] == 1:
                pos_correct_count +=1
            elif labels[i] == 0:
                neg_correct_count +=1
    print('correct for 1 class:', pos_correct_count)
    print('correct for 0 class:', neg_correct_count)

    print('accuracy: ', round(accuracy(preds, labels), 4))
    print('precision: ', round(precision(preds, labels), 4))
    print('recall: ', round(recall(preds, labels), 4))
    print('F1: ', f1_score(labels, preds))

def remove_spaces(string_):
    return ' '.join(string_.split())

def call_chatgpt(Model, sys_prompt, user_prompt, user_model):

    completion_result = Model.create(model=user_model,
                                    messages=[
    {"role": "system", "content": sys_prompt}, \
    {"role": "user", "content": user_prompt}
                                                        ],
                                    )
    time.sleep(1)

    return completion_result


def prompt_design():
    pass


        
def myDataProcess(dataFile):

    df = pd.read_csv(str(dataFile), encoding='UTF-8')

    # shuffle the dataframe
    df_shuffled = df.sample(frac=1).reset_index(drop=True)


    df_shuffled["new_message1"].apply(lambda x: x.replace('<enter>', '$enter').replace('<tab>', '$tab'). \
                                    replace('<url>', '$url').replace('<version>', '$version') \
                                    .replace('<pr_link>', '$pull request>').replace('<issue_link >',
                                                                                    '$issue') \
                                    .replace('<otherCommit_link>', '$other commit').replace("<method_name>",
                                                                                            "$method") \
                                    .replace("<file_name>", "$file").replace("<iden>", "$token"))

    whyLabels = df_shuffled['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (1 if x == 2.0 else (0 if x == 3.0 else 1))))
    whatLabels = df_shuffled['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (1 if x == 3.0 else 0))))
    Labels = df_shuffled['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (0 if x == 3.0 else 1))))
    print("load data successfully!")
    messages = list(df_shuffled['new_message1'])

    return messages, (whyLabels), (whatLabels), (Labels)

def prompt_design_no_context(message):
    # prompt reference: https://arxiv.org/pdf/2103.12407.pdf
    # prompt should be polished on the validation dataset
    return 'A good git commit message should summarize the changes in this commit and describe the reasons for the changes.\n\
Is the following commit a good git commit message? Answer only yes or no.\n\n'+message

def votes(results):
    votes = []
    if str(results.choices[0]['message']['content']) == 'Yes.':
        votes.append(1)
    else:
        votes.append(0)
    if str(results.choices[1]['message']['content']) == 'Yes.':
        votes.append(1)
    else:
        votes.append(0)
    if str(results.choices[2]['message']['content']) == 'Yes.':
        votes.append(1)
    else:
        votes.append(0)
    array = np.array(votes)

    # Count the occurrences of each element
    counts = np.bincount(array)

    # Get the indices of the most frequent elements
    most_common_indices = np.where(counts == counts.max())[0]

    # Display the most common elements
    # for element in most_common_indices:
        # print(f"Element: {element}, Count: {counts[element]}")

    if len(most_common_indices) == 1:
        return most_common_indices[0]
    else:
        print('error')
        exit()

def example_API_documentation():
    
    original_API_doc = ''' 
[[original API documentation]]:
<This API is android.database.sqlite.SQLiteDatabase.onCreate in Android library.\n
[[[Function]]]: Called when the database is created for the first time. This is where the creation of tables and the initial population of the tables should happen.\n
[[[Parameter]]]: db	SQLiteDatabase: The database.\n
[[[Notes]]]: None>
'''
    extractive_summaries = '''
Below are [[extractive summaries of augmented API document sections]].\n\
<[[[Extractive summaries for Function section]]]:  onCreate is responsible for creating your tables inside your SQLite database, and onUpgrade will be responsible for upgrading your database if you do make any changes. 2)onUpgrade() This method called when we change the database version,then this methods gets invoked.It is used for the alter the table structure like adding new column after creating DB Schema. If you want the onUpgrade() method to be called, you need to increment the version number in your code. Increment the database version so that onUpgrade() is invoked. \n
[[[Extractive summaries for Parameter section]]]: I'm just going to create the table and its structure.  To do that, I'll remove the TODO comment, and I'll use the database argument that's being passed in.  It's named db, and I'll call a method called execute SQL or execSQL for short, and I'll pass in my constant that contains the SQL command that will create the table, that'll be TABLE_CREATE. The version number is the int argument passed to the [constructor (hyper-link)]. In the database file, the version number is stored in [PRAGMA user_version (hyper-link)]. So when the database helper constructor is called with a name (2nd param), platform checks if the database exists or not and if the database exists, it gets the version information from the database file header and triggers the right call back.\n
[[[Extractive summaries for Notes section]]]:  If the database already exists, but I've indicated through the database version value that I'm changing the version, that is that I've incremented it, then the onUpgrade method will be called. Below explanation explains onUpgrade case with an example. Example pseudo code below: In the onCreate method, we're going to start by creating a string to build the weather entry table using data defined within the weather entry contract. In the onCreate method, you should add code that creates your database tables and if you like, you can also add code to add data.\n>
'''

    augmented_API_doc = '''
Below are [[Augmented API documentation]].\n
<[[[Function]]]: 
Called when the database is created for the first time. This is where the creation of tables and the initial population of the tables should happen.\n    
It's important to note that the onCreate() method will not be called if the code is changed and the application is relaunched in the emulator. Once the onCreate() method has been executed during the initial deployment, it will not be called again in subsequent launches.
The onCreate() method is triggered when getWritableDatabase() or getReadableDatabase() is called on the database helper and the database file does not exist. However, if the database file already exists and the version number matches the requested one, no callback such as onCreate() will be invoked.

[[[Parameter]]]: db	SQLiteDatabase: The database.\n
If onCreate() returns successfully (doesn't throw an exception), the database is assumed to be created with the requested version number. 

[[[Notes]]]: None\n
The onCreate() method is invoked only when the database file doesn't exist, so there is no need to use the DROP TABLE command. This suggests that the API handles the creation of the database file automatically when it doesn't exist.
To execute a query, it needs to be executed within the onCreate callback method, which requires extending a class with the SQLiteOpenHelper class. The query can be executed directly by using db.execSQL() with the entire query as the parameter.
Inside the SQLiteOpenHelper class, the user creates the entire database structure. It includes methods such as onCreate, onUpgrade, and queries for insert, update, and delete operations. This class acts as a subclass of SQLiteOpenHelper.
If the database already exists, but the version has been incremented, the onUpgrade method will be called. This suggests that the API handles versioning of the database and provides a mechanism for upgrading the database schema when necessary.
If the user needs to recreate the table after dropping it, they can simply call the onCreate method. However, this is an exception to the general practice of not directly calling the onCreate method.>
'''

    return original_API_doc + extractive_summaries + augmented_API_doc
    

def get_ground_truth(groundtruth, idxs):
    return [groundtruth[i] for i in idxs]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def sum_no_extract_no_context(model, Model, function, parameter, others, eval_api):

    if eval_api == 'addToBackStack' or eval_api == 'dispatchTouchEvent':
        lib = 'android'
    else:
        lib = 'pytorch'

    sys_prompt = 'You are an abstractive summarizer that follows the output pattern.'
    user_prompt = 'Please augment the API documentation based on following format instruction.\n\
    [Format instruction]:\n\
    Each API documentation describes the function, parameter, and notes sections of an API.\n\
    Function information includes but not limited to the executive summary, expected behavior, state information and transitions, defined algorithms, and cause of exceptions of the API.\n\
    Parameter information includes but not limited to the range of valid argument values, return values, and behaviors of passing an invalid argument to the API.\n\
    Notes information includes but not limited to OS/hardware dependencies, allowed implementation variances, security constraints, and references to any external specifications of the API, also other insightful tips (e.g., API performance, legal concern).\n\
    Please augment the API documentation by adding additional content to each section while preserving the original content (i.e., supplementing each section of api documentation with useful and factual information defined above) within at most five sentences.\n\
    You are required to choose essential information from each section rather than summarizing all the relevant details.\n\
    You\'re allowed to summarize information from external resources.\n\
    [API documentation]:\n\
    This API is <'+eval_api+'> in '+lib+' library.\n\
    Function: '+function+'\n\
    Parameter: '+parameter+'\n\
    Notes: '+others+'\n'

    # print(user_prompt)
    # exit()

    # try:
    results = call_chatgpt(Model, sys_prompt, user_prompt, model)
    logging.info(f'prompt: {sys_prompt+user_prompt}')
    for item in results.choices:
        logging.info(f'result: {item["message"]["content"]}')
    # except:
        # print('error')
    return item["message"]["content"]

def sum_no_extract_but_context(model, Model, function, parameter, others, eval_api, so_answer_format, video_caption_format):

    if eval_api == 'addToBackStack' or eval_api == 'dispatchTouchEvent':
        lib = 'android'
    else:
        lib = 'pytorch'

    sys_prompt = 'You are an abstractive summarizer that follows the output pattern.'
    user_prompt = 'Please augment the API documentation based on following format instruction and external resources.\n\
[Format instruction]:\n\
Each API documentation describes the function, parameter, and notes sections of an API.\n\
Function information includes but not limited to the executive summary, expected behavior, state information and transitions, defined algorithms, and cause of exceptions of the API.\n\
Parameter information includes but not limited to the range of valid argument values, return values, and behaviors of passing an invalid argument to the API.\n\
Notes information includes but not limited to OS/hardware dependencies, allowed implementation variances, security constraints, and references to any external specifications of the API, also other insightful tips (e.g., API performance, legal concern).\n\
Please augment the API documentation by adding additional content to each section after the original content (i.e., supplementing each section of api documentation with useful and factual information defined above) within at most five sentences.\n\
You are required to choose key information from each section rather than summarizing all the relevant details.\n\
You\'re allowed to summarize information from external resources.\n\
[API documentation]:\n\
This API is <'+eval_api+'> in '+lib+' library.\n\
Function: '+function+'\n\
Parameter: '+parameter+'\n\
Notes: '+others+'\n\
Below are the external resources that you can refer to.\n\
[Related Stack Overflow Questions and Answers]:\n'+so_answer_format+'\n\
[Related Youtube Videos]:\n'+video_caption_format+'\n'

    # consider the api call head 
    while num_tokens_from_string(sys_prompt+user_prompt, "cl100k_base")>8179:
        # set 100 for faster speed
        if num_tokens_from_string(sys_prompt+user_prompt, "cl100k_base")>12000:
            user_prompt = user_prompt[:-1000]
        else:
            user_prompt = user_prompt[:-10]


    print('finished cutting off')

    # try:
    results = call_chatgpt(Model, sys_prompt, user_prompt, model)
    logging.info(f'prompt: {sys_prompt+user_prompt}')
    for item in results.choices:
        logging.info(f'result: {item["message"]["content"]}')
    # except:
        # print('error')
    return item["message"]["content"]


def sum_extract_and_context(model, Model, function, parameter, others, eval_api, so_answer_format, video_caption_format, function_sum, parameter_sum, others_sum):

    if eval_api == 'addToBackStack' or eval_api == 'dispatchTouchEvent':
        lib = 'android'
    else:
        lib = 'pytorch'

    sys_prompt = 'You are an abstractive summarizer that follows the output pattern.'
    user_prompt = 'Please generate the [augmented API documentation] based on following [format instruction], [original API documentation], [example of generating augmented API documentation], [extractive summaries of augmented API document sections], and [external resources].\n\
========================\n\
[Format instruction]:\n\
Each API documentation describes the function, parameter, and notes sections of an API.\n\
Function information includes but not limited to the executive summary, expected behavior, state information and transitions, defined algorithms, and cause of exceptions of the API.\n\
Parameter information includes but not limited to the range of valid argument values, return values, and behaviors of passing an invalid argument to the API.\n\
Notes information includes but not limited to OS/hardware dependencies, allowed implementation variances, security constraints, and references to any external specifications of the API, also other insightful tips (e.g., API performance, legal concern).\n\
Please augment the API documentation by adding additional content to each section after the original content (i.e., supplementing each section of api documentation with useful and factual information defined above) within at most five sentences.\n\
The [extractive summaries of augmented API document sections] refers to the valuable additional information to original document\'s sections.\
You\'re required to augment the [API documentation] by selecting valuable information from [extractive summaries of augmented API document sections] and touch-up them.\n\
One example of generating augmented API documentation is shown in [example of generating augmented API documentation].\n\
You are required to choose valuable information from each section rather than summarizing all the relevant details.\n\
========================\n\
[original API documentation]:\n\
This API is <'+eval_api+'> in '+lib+' library.\n\
Function: '+function+'\n\
Parameter: '+parameter+'\n\
Notes: '+others+'\n\
========================\n\
Below are [example of generating augmented API documentation].\n\
'+example_API_documentation()+'\n\
========================\n\
Below are [extractive summaries of augmented API document sections].\n\
This API is <'+eval_api+'> in '+lib+' library.\n\
[[[Extractive summaries for Function section]]]: '+function_sum+'\n\n\
[[[Extractive summaries for Parameter section]]]: '+parameter_sum+'\n\n\
[[[Extractive summaries for Notes section]]]: '+others_sum+'\n\n\
========================\n\
Below are the [external resources] that you can refer to.\n\
[Related Stack Overflow Questions and Answers]:\n'+so_answer_format+'\n\
[Related Youtube Videos]:\n'+video_caption_format+'\n'

    # consider the api call head 
    while num_tokens_from_string(sys_prompt+user_prompt, "cl100k_base")>8179:
        # set 100 for faster speed
        if num_tokens_from_string(sys_prompt+user_prompt, "cl100k_base")>12000:
            user_prompt = user_prompt[:-1000]
        else:
            user_prompt = user_prompt[:-10]


    print('finished cutting off')

    # try:
    results = call_chatgpt(Model, sys_prompt, user_prompt, model)
    logging.info(f'prompt: {sys_prompt+user_prompt}')
    for item in results.choices:
        logging.info(f'result: {item["message"]["content"]}')
    # except:
        # print('error')
    return item["message"]["content"]


def get_api_doc(api_path, eval_api):
    with open(os.path.join(api_path, eval_api,'function.txt'), 'r') as f:
        function = f.read()
    with open(os.path.join(api_path, eval_api,'parameter.txt'), 'r') as f:
        parameter = f.read()
    with open(os.path.join(api_path, eval_api,'others.txt'), 'r') as f:
        others = f.read()

    return function, parameter, others

def so_format(reader):
    answer = []
    count = 0
    for row in reader:
        if not row[2].startswith('Paragraph end'):
            answer.append(row[2])
        else:
            answer.append('\n')

        if count == 0:
            question_title = row[3]
            question_body = row[-1]
        count+=1

    # print(answer, question_title, question_body)
    return answer, question_title, question_body

def run():

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create a stream handler to capture the print output
    class StreamHandlerWrapper(logging.StreamHandler):
        def emit(self, record):
            msg = self.format(record)
            logger.info(msg)

    # Add the stream handler to the logger
    logger.addHandler(StreamHandlerWrapper(sys.stdout))

    logging.basicConfig(filename='script.log', filemode='w',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='gpt labeling')
    
    # ['no_extract_no_context', 'no_extract_context', 'extract_no_context', 'extract_context']
    # extract: based on extractive summarization
    # context: API-related external resource
    parser.add_argument('--mode', type=str, help='Set the mode of the script.')
    parser.add_argument('--so', default='8', help='set the number of Stack Overflow answers')
    parser.add_argument('--video', default='3', help='set the number of videos')



    args = parser.parse_args()

    """
    Call GPT4   
    """

    print('start calling GPT4. Note that You need to set the API key in the script.')

    key = 'YOUR_API_KEY'
    openai.api_key = key
    Model = openai.ChatCompletion()
    model = 'gpt-4'

    # Load data
    api_list = ['']
    api_path = '//APISummarization/data/API_Document/End2endEvaluation'
    external_path = '//APISummarization/data/so_api_relevant_answer_info/labeling/first_stage/all'
    extsum_path = '//APISummarization/src/classification/result/'
    # data_path = '//APISummarization/data/labeling/collection/'
    results_path = '//APISUM_replication/src/GPT/results/no_extract_no_context'

    for eval_api in os.listdir(api_path):

        if args.mode == 'no_extract_no_context':
            # input: API documentation and general prompts

            # extract API documentation
            function, parameter, others = get_api_doc(api_path, eval_api)

            # summarization
            updated_sum = sum_no_extract_no_context(model, Model, function, parameter, others, eval_api)

            # save the updated API documentation
            if not os.path.exists(os.path.join(results_path,eval_api)):
                os.makedirs(os.path.join(results_path,eval_api))
            with open(os.path.join(results_path,eval_api, 'updated_sum_incontext.txt'), 'w') as f:
                f.write(updated_sum)

        if args.mode == 'no_extract_but_context':
            # extract API documentation
            function, parameter, others = get_api_doc(api_path, eval_api)

            so_answer_format = ''
            video_caption_format = ''

            # extract the external resources
            smaller_number = lambda x, y: x if x < y else y
            # control the number of SO answers
            len_so_answers = len(os.listdir(os.path.join(external_path, eval_api,'so')))
            # generate the random index
            number_of_so_answers = smaller_number(len_so_answers, int(args.so))
            idxs = np.random.choice(len_so_answers, number_of_so_answers, replace=False)

            # control the number of YouTube videos
            len_video = len(os.listdir(os.path.join(external_path, eval_api,'video')))
            # generate the random index
            number_of_videos = smaller_number(len_video, int(args.video))
            idxs_video = np.random.choice(len_video, number_of_videos, replace=False)

            for index, external_page in enumerate(os.listdir(os.path.join(external_path, eval_api,'so'))):
                # extract SO answers
                if external_page.startswith('answer'):
                    with open(os.path.join(external_path, eval_api,'so',external_page), 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        answer, question_title, question_body = so_format(reader)

                if index in idxs:
                # SO answers format
                    so_answer_format += '[[Question and Answer '+str(index)+' ]]\n\
                    [[[question title]]]: '+replace_whitespace_with_space(question_title)\
                                +'\n[[[question body]]]: '+replace_whitespace_with_space(question_body)\
                                +'\n[[[answer]]]: '+replace_whitespace_with_space('\n'.join(answer))\
                                +'\n[[end]]\n'
            
            # print(so_answer_format)

            # extract youtube info
            for index, external_page in enumerate(os.listdir(os.path.join(external_path, eval_api,'video'))):

                if index in idxs_video:
                    if external_page.endswith('.txt'):
                        with open(os.path.join(external_path, eval_api,'video',external_page), 'r') as f:
                            video_info = f.read()
                        result = video_info.replace('\n', ' ')
                        video_caption_format += '[[video caption]]: '+result+'\n[[[end]]]\n'

            # print(video_caption_format)
            # exit()

            # summarization
            updated_sum = sum_no_extract_but_context(model, Model, function, parameter, others, eval_api, so_answer_format, video_caption_format)

            # save the updated API documentation
            if not os.path.exists(os.path.join(results_path,eval_api)):
                os.makedirs(os.path.join(results_path,eval_api))
            with open(os.path.join(results_path,eval_api, 'updated_sum_no_extract_but_context.txt'), 'w') as f:
                f.write(updated_sum)

        if args.mode == 'extract_and_context':
            # extract API documentation
            function, parameter, others = get_api_doc(api_path, eval_api)

            so_answer_format = ''
            video_caption_format = ''

            # extract the external resources
            smaller_number = lambda x, y: x if x < y else y
            # control the number of SO answers
            len_so_answers = len(os.listdir(os.path.join(external_path, eval_api,'so')))
            # generate the random index
            number_of_so_answers = smaller_number(len_so_answers, int(args.so))
            idxs = np.random.choice(len_so_answers, number_of_so_answers, replace=False)

            # control the number of YouTube videos
            len_video = len(os.listdir(os.path.join(external_path, eval_api,'video')))
            # generate the random index
            number_of_videos = smaller_number(len_video, int(args.video))
            idxs_video = np.random.choice(len_video, number_of_videos, replace=False)

            for index, external_page in enumerate(os.listdir(os.path.join(external_path, eval_api,'so'))):
                # extract SO answers
                if external_page.startswith('answer'):
                    with open(os.path.join(external_path, eval_api,'so',external_page), 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        answer, question_title, question_body = so_format(reader)

                if index in idxs:
                # SO answers format
                    so_answer_format += '[[Question and Answer '+str(index)+' ]]\n\
                    [[[question title]]]: '+replace_whitespace_with_space(question_title)\
                                +'\n[[[question body]]]: '+replace_whitespace_with_space(question_body)\
                                +'\n[[[answer]]]: '+replace_whitespace_with_space('\n'.join(answer))\
                                +'\n[[end]]\n'
            
            # print(so_answer_format)

            # extract youtube info
            for index, external_page in enumerate(os.listdir(os.path.join(external_path, eval_api,'video'))):

                if index in idxs_video:
                    if external_page.endswith('.txt'):
                        with open(os.path.join(external_path, eval_api,'video',external_page), 'r') as f:
                            video_info = f.read()
                        result = video_info.replace('\n', ' ')
                        video_caption_format += '[[video caption]]: '+result+'\n[[[end]]]\n'

            # extract extractive information
            sum_path = '//APISummarization/src/summarization/our_approach/biased_textrank/result'
            with open(os.path.join(sum_path, eval_api, 'function_nobias.txt'), 'r') as f:
                function_sum = f.read().replace('\n', ' ')
            with open(os.path.join(sum_path, eval_api, 'others_nobias.txt'), 'r') as f:
                others_sum = f.read().replace('\n', ' ')            
            with open(os.path.join(sum_path, eval_api, 'parameter_nobias.txt'), 'r') as f:
                parameter_sum = f.read().replace('\n', ' ')

            # summarization
            updated_sum = sum_extract_and_context(model, Model, function, parameter, others, eval_api, so_answer_format, video_caption_format, function_sum, parameter_sum, others_sum)

            # save the updated API documentation
            if not os.path.exists(os.path.join(results_path,eval_api)):
                os.makedirs(os.path.join(results_path,eval_api))
            with open(os.path.join(results_path,eval_api, 'updated_sum_extract_and_context.txt'), 'w') as f:
                f.write(updated_sum)






if __name__ == '__main__':
    run()