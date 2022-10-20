import pickle
import os
from platform import python_branch
from youtube_transcript_api  import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from collections import defaultdict


def get_video_info_from_pickle(video_path):
    video_infos = []
    for video_meta in os.listdir(video_path):
        with open(os.path.join(video_path, video_meta), 'rb') as f:
            video_info = pickle.load(f)
            video_infos.append(video_info)
    return video_infos

def get_youtube_captions(video_id, languages=None,filter='mau'):
    """
    Use YouTube's API to pull captions from a video.

    @param youtube_video_url: String holding the youtube link.
    @param languages:         Language on which captions must be downloaded.

    Note: The semantic similarity method only works for english, as the model
    trained by this script is trained on the English Brown Corpus.

    The fuzzy similarity method works across all languages.

    """
    if languages is None:
        languages = ['en']

    captions_and_timestamps = dict()

    if filter =='mau':
        # filter for manually created transcripts
        try:
            if YouTubeTranscriptApi.list_transcripts(video_id):
                mua_transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_manually_created_transcript(['en'])
                if mua_transcript:
                    print('Found manually created transcript for video %s' % video_id); 
                    transcript = mua_transcript.fetch()
                    formatter = TextFormatter()

                    # .format_transcript(transcript) turns the transcript into a JSON string.
                    txt_formatted = formatter.format_transcript(transcript)
                    # print(txt_formatted)
                    
                    # write to txt file
                    with open('/storage/chengran/APISummarization/data/video_api_info/caption_data/test/manua_data/%s.txt' % video_id, 'w') as f:
                        f.write(txt_formatted) 
                    

            else:
                print('No transcript found for video %s' % video_id)

        except:
            return

    elif filter =='auto':
        # filter for generated  transcripts
        try:
            if YouTubeTranscriptApi.list_transcripts(video_id):
                gen_transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_generated_transcript(['en'])
                if gen_transcript:
                    print('Found generated created transcript for video %s' % video_id); 
                    transcript = gen_transcript.fetch()
                    formatter = TextFormatter()

                    # .format_transcript(transcript) turns the transcript into a JSON string.
                    txt_formatted = formatter.format_transcript(transcript)
                    with open('/storage/chengran/APISummarization/data/video_api_info/caption_data/test/generated_data/%s.txt' % video_id, 'w') as f:
                        f.write(txt_formatted) 

            else:
                print('No transcript found for video %s' % video_id)

        except:
            return


if __name__ == '__main__':
    video_path = '/storage/chengran/APISummarization/data/video_api_info/Meta_Data/Test'
    video_infos = get_video_info_from_pickle(video_path)
    # print(video_infos[0][0])
    for num,video in enumerate(video_infos[0]):
        video_id = video['id']['videoId']
        captions = get_youtube_captions(video_id, languages=None,filter='auto')
        




