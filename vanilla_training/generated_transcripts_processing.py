##########
# Once you got the generated transcriptions, you have to process them,
# since the reference transcripts have some special characteristics,
# such as they user words to represent numbers, no filler words like "Um"
# example:
# Generated transcription:  Um, it's 20.
# Reference transcription: it's twenty
# they are generally the same, I would say the WER for this case is 0%
##########

import re
import inflect

p = inflect.engine()

def convert_numbers_to_words(text):
    # Replace ordinal numbers (like 14th) with their word equivalents
    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', lambda x: p.number_to_words(int(x.group(1))) + x.group(2), text)
    
    # Replace cardinal numbers (like 2000) with their word equivalents
    text = re.sub(r'\b\d+\b', lambda x: p.number_to_words(int(x.group())), text)
    
    return text

def process_generated_transcriptions(generated_transcriptions):
    # 1. Remove "Um", "um"
    generated_transcriptions = [re.sub(r'\bum\b', '', x, flags=re.IGNORECASE) for x in generated_transcriptions]

    # 2. Remove "Uh", "uh"
    generated_transcriptions = [re.sub(r'\buh\b', '', x, flags=re.IGNORECASE) for x in generated_transcriptions]

    # 3. translate all numbers to english words
    generated_transcriptions = [convert_numbers_to_words(x) for x in generated_transcriptions]

    return generated_transcriptions