##########
# Once you got the generated transcriptions, you have to process them,
# since the reference transcripts have some special characteristics,
# such as they user words to represent numbers, no filler words like "Um"
# example:
# Generated transcription:  Um, it's twenty.
# Reference transcription: it's 20
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
    # 1. Remove "Um", "um"
    generated_transcriptions = [re.sub(r'\bum\b', '', x, flags=re.IGNORECASE) for x in generated_transcriptions]

    # 2. Remove "Uh", "uh"
    generated_transcriptions = [re.sub(r'\buh\b', '', x, flags=re.IGNORECASE) for x in generated_transcriptions]

    # 3. translate all numbers to english words
    generated_transcriptions = [convert_numbers_to_words(x) for x in generated_transcriptions]

    return generated_transcriptions