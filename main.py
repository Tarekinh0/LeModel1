import os

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

import detoxify
import nltk

from bark import SAMPLE_RATE
import numpy
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

import warnings
warnings.filterwarnings("ignore")

## Empty GPY cache
torch.cuda.empty_cache()

# init env variables

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# init needed models

context = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, \
and you only answer questions related to computer science. For politically sensitive questions, \
security and privacy issues, and other non-computer science questions, you will refuse to answer."


model_name_or_path = "TheBloke/deepseek-coder-1.3b-instruct-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main",
                                             local_files_only=True).cuda()

model.save_pretrained("./libs/saved/deepseeker")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, local_files_only=True)
tokenizer.save_pretrained("./libs/saved/deepseeker")

# Defining the functions we will use

def detox(seq):
    results = detoxify.Detoxify('original-small', device='cuda').predict(seq)
    for k, v in results.items():
        if v > 0.1:
            return f"Error: Sorry but your prompt/answer did not pass the {k} test."
    torch.cuda.empty_cache()
    return "OK"

def vocal_answer(answer):
    sentences = nltk.sent_tokenize(answer)
    SILENCE = numpy.zeros(int(0.25*SAMPLE_RATE))  # quarter second of silence
    SILENCE = SILENCE.reshape(-1,1)
    pieces = [] 
    synthesiser = pipeline("text-to-speech", "suno/bark-small")
    # Breaking the response into pieces to generate independantly
    for sentence in sentences:
        speech = synthesiser(sentence, forward_params={"do_sample": True, "pad_token_id": 0})
        pieces += [speech['audio'].T, SILENCE.copy()]
        torch.cuda.empty_cache()
    final_speech = numpy.row_stack(pieces)
    write_wav("response.wav", rate=speech["sampling_rate"], data=final_speech)
    Audio("response.wav")


def generate_format_output(prompt_template):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=255,
        do_sample=True,
        temperature=0.7,
        top_p=0.1,
        top_k=35,
        repetition_penalty=1.1
    )
    output = pipe(prompt_template)[0]['generated_text']
    output = output.split("Response:")[1]
    # output = output.split("\n")[0]
    torch.cuda.empty_cache()
    return output

def loop():
    sequence = ""
    while (sequence != "exit"):
        # Get prompt input
        sequence = input("You: ")
        if sequence == 'exit': break
        # Detox the input
        detox_response = detox(sequence)
        if detox_response != "OK":
            print(detox_response)
            continue
        # We accept the prompt
        prompt_template= f"{context} ### Instruction: {sequence} ### Response:"
        # generate the answer
        answer = generate_format_output(prompt_template)
        detox_response = detox(answer)
        if detox_response != "OK":
            print(detox_response)
            continue
        print(f"Assistant: {answer}")
        vocal_answer(answer)


print("\n\nWrite \'exit\' as a prompt to exit the program !\n\n")

vocal_answer("Testing if the voice AI works well. By the way, how ARE you ?")
# loop()



def vocal_answer(answer):
    sentences = nltk.sent_tokenize(answer)
    SILENCE = numpy.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
    SPEAKER = "v2/en_speaker_6"
    pieces = []
    for sentence in sentences:
        synthesiser = pipeline("text-to-speech", "suno/bark-small")
        speech = synthesiser(sentence, forward_params={"do_sample": True, "pad_token_id": 0})
        pieces += [speech, SILENCE.copy()]
        torch.cuda.empty_cache()
    final_speech = numpy.concatenate(pieces)
    write_wav("response.wav", rate=speech["sampling_rate"], data=final_speech.T)
    # read_wav("response.wav")