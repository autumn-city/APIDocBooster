# call for SO API
As searching for stack overflow cost too much time than my expectation, i turn to use the api offered by SO.
## the API authentication info
https://stackapps.com/apps/oauth/view/24424

The api key is `app_key=tFaahBz1)Kq70INbmCkYrw((` .


# Search Algorith
## initialization
+ exact match
It doesn't work well as the 
# Query
select the APIs that have math formulation, the reason is that those APIs are more likely to require augmentation by external resources.
# query involves "Pytorch"
I found that the number of questions is relatively small (less than 10).
But if I search without the pytorch keyword, the result may not make sense as some of the api names look similar to everyday words.
However, collecting the related answer as much as possible is the highest priority. 
Thus I decided to use the API name as the query and omit the keyword pytorch.
Then in the benchmark building phase, we can filter out and throw those APIs whose name has ambiguity.

# Dataset
the data dump is not the latest so that the some of the search result from SO API can't be find in current data dump
Just ignore it now (for observation), will update the data dump before the actual labeling. 

# Extraction
## method
case insensitive exact match between the API and each sentence in the answer.
## bad case
some bad case makes the performance of sent spliter poor, like e.g., i.e.,.
I've fixed


# heuristic filter
+ 手动把所有相关的问题给过滤一遍
    + 相关的问题的所有答案都是和API相关的
+ 手动过滤句子 (api info)

baseline 过滤相关问题的过程
能否在方法里自动化成一个分类问题

# 修改输入
