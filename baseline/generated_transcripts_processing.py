##########
# Once you got the generated transcriptions, you have to process them,
# since the reference transcripts have some special characteristics,
# such as they don't have punctuation, no filler word like "Um", all lowercase
# example:
# Generated transcription:  Um, I don't remember.
# Reference transcription: i don't remember
# they are generally the same, I would say the WER for this case is 0%
##########

import re
import inflect

p = inflect.engine()

def convert_numbers_to_words(text):
    # Split the text into words
    words = text.split()

    # Convert each word to its corresponding word form if it's a digit
    converted_words = [p.number_to_words(word) if word.isdigit() else word for word in words]

    # Join the words back into a single string
    return ' '.join(converted_words)

def process_generated_transcriptions(generated_transcriptions):
    """
    Process the generated transcriptions to match the characteristics of reference transcripts.

    Parameters:
    all_generated_transcriptions (list): A list of generated transcription strings.

    Returns:
    list: A list of processed transcription strings.
    """
    # 1. Remove "Um", "um"
    generated_transcriptions = [re.sub(r'\bum\b', '', x, flags=re.IGNORECASE) for x in generated_transcriptions]

    # 2. Remove "Uh", "uh"
    generated_transcriptions = [re.sub(r'\buh\b', '', x, flags=re.IGNORECASE) for x in generated_transcriptions]
    
    # 3. Remove all punctuations except apostrophes
    # generated_transcriptions = [re.sub(r"[^\w\s']", '', x) for x in generated_transcriptions]

    # 4. translate all numbers to english words
    generated_transcriptions = [convert_numbers_to_words(x) for x in generated_transcriptions]
    
    # 5. Convert to lowercase
    # generated_transcriptions = [x.lower() for x in generated_transcriptions]
    
    # 6. Strip extra spaces
    generated_transcriptions = [re.sub(r'\s+', ' ', x.strip()) for x in generated_transcriptions]

    return generated_transcriptions