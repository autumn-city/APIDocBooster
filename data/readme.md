# Intro
This folder contains the data of APISUMBench, which is used for training and evaluating the APIDOCBooster.
## Classification_Stage
Each data is a sentence extracted from Stack Overflow answer posts or YouTube videos and associated with a specific API. 
## Summarization_Stage
Each sub-folder is the API name. Then for each API, there are four `.txt` file:
```
doc.txt: in which the content refers to the original API documentation;
function.txt: which is the summary of functionality section for this API;
parameter.txt: which is the summary of parameter section for this API;
others.txt: which is the summary of notes section for this API;
```