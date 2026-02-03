import pandas as pd


speeches_prompt =  """Generate a prompt by a user whose answer is the following speech. The prompt can ask about the topic of the speech or 
 Example: 
 Speech: {speech}
 Prompt: Hi Obama, generate an inspiring speech on education for students. 
 """



def generate_prompts_for_speeches(speeches):
    with open("education_example.txt", "r") as file:
        example = file.readlines()
        prompt = speeches_prompt.format(speech=example)



def generate_generic_responses()