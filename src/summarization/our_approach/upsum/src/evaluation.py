from lib2to3.pgen2.literals import test
from pyrouge import Rouge155
import os
import shutil
import re
import logging
import util.rouge_evaluator as rouge_wrapper
from numpy import mean
import pickle


def inter_aggrement():
    '''
    calculate the rouge score for the answerbot output to the groundtruth
    '''
    evaluator = rouge_wrapper.RougeEvaluator(                    

                    # system_filename_pattern='([\s\S]*)_[\s\S]*.txt',
                    system_filename_pattern='([\s\S]*)_[\s\S]*.txt',

                    model_filename_pattern='#ID#.txt',
                    system_dir=os.path.join('../result'),
                    model_dir=os.path.join(documents_dir)

                       )
    output_1 = evaluator.evaluate()
    # print(output_1['short_output'])
    print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator.short_result(output_1))

def inter_aggrement_API(api,sys_dir):
    '''
    calculate the rouge score for the answerbot output to the groundtruth
    '''
    evaluator = rouge_wrapper.RougeEvaluator(                    

                    system_filename_pattern='([\s\S]*)_[\s\S]*.txt',
                    # system_filename_pattern='(parameter)_[\s\S]*.txt',

                    model_filename_pattern='#ID#.txt',
                    # system_dir=os.path.join('../../../baseline_result/',api),
                    system_dir=sys_dir,
                    # system_dir=os.path.join('../result/',api),
                    model_dir=os.path.join(documents_dir,api)

                       )
    output_1 = evaluator.evaluate()
    # print(output_1['short_output'])
    # print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator.short_result(output_1))
    # print(api)
    return evaluator.short_result(output_1)



if __name__ == '__main__':
    documents_dir = '/storage/chengran/APISummarization/data/labeling/collection'
    ablation_dir = '/storage/chengran/APISummarization/src/summarization/our_approach/ablation_result'
    # documents_dir = '/storage/chengran/APISummarization/src/summarization/groundtruth'
    rouge_1_result = []
    rouge_2_result = []
    rouge_l_result = []

    overall = {}


    for index, file in enumerate(os.listdir(documents_dir)): 
        temp = inter_aggrement_API(file, os.path.join('../../ablation_result/0.7_0.8',file))
        # temp = inter_aggrement_function(file)
        rouge_1_result.append(float(temp[0]))
        rouge_2_result.append(float(temp[1]))
        rouge_l_result.append(float(temp[2]))

    print('the final roug-1 score is : ',mean(rouge_1_result))
    print('the final roug-2 score is : ',mean(rouge_2_result))
    print('the final roug-l score is : ',mean(rouge_l_result))

    # ablation study
    # for ablation in os.listdir(ablation_dir):
    #     for index, file in enumerate(os.listdir(documents_dir)): 
    #         # if file == 'onPause':
    #             # continue
    #         # print('currently working on the API: ',file)
    #         sys_dir = os.path.join(ablation_dir,ablation,file)
    #         temp = inter_aggrement_API(file,sys_dir)
    #         # temp = inter_aggrement_function(file)
    #         rouge_1_result.append(float(temp[0]))
    #         rouge_2_result.append(float(temp[1]))
    #         rouge_l_result.append(float(temp[2]))
    #     print('the setting is', os.path.join(ablation_dir,ablation))
    #     print('the final roug-1 score is : ',mean(rouge_1_result))
    #     print('the final roug-2 score is : ',mean(rouge_2_result))
    #     print('the final roug-l score is : ',mean(rouge_l_result))
    #     print('\n\n\n\n')

    #     overall[ablation]=[mean(rouge_1_result),mean(rouge_2_result),mean(rouge_l_result)]

    # print(overall)

    with open('../../filename.pickle', 'wb') as handle:
        pickle.dump(overall, handle, protocol=pickle.HIGHEST_PROTOCOL)


