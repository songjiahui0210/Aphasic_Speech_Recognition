##########
# Once you got the generated transcriptions, you have to process them,
# since the groud truth transcripts have some special characteristics,
# such as they don't have punctuation, no filler word like "Um", all lowercase
##########

import re

def process_generated_transcriptions(generated_transcriptions):
    """
    Process the generated transcriptions to match the characteristics of ground truth transcripts.

    Parameters:
    all_generated_transcriptions (list): A list of generated transcription strings.

    Returns:
    list: A list of processed transcription strings.
    """
    # 1. Remove "Um"
    generated_transcriptions = [re.sub(r'Um', '', x) for x in generated_transcriptions]
    
    # 2. Remove all punctuations except apostrophes
    generated_transcriptions = [re.sub(r"[^\w\s']", '', x) for x in generated_transcriptions]
    
    # 3. Convert to lowercase
    generated_transcriptions = [x.lower() for x in generated_transcriptions]
    
    # 4. Strip extra spaces
    generated_transcriptions = [x.strip() for x in generated_transcriptions]
    generated_transcriptions = [re.sub(r'\s+', ' ', x) for x in generated_transcriptions]

    return generated_transcriptions