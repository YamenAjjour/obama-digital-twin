import pandas as pd
import os
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from time import sleep
# Only run this block for Gemini Developer API


load_dotenv()

generate_prompt =  """Generate a brief generic prompt by a user whose answer is the following document.
 Only output the prompt and nothing else. Make sure to maximally write two sentences !
 The prompt can point to the topic of the speech and the audience.  
 Example: 
 Speech: {speech}
 Prompt: Hi Obama, generate an inspiring speech on education for students. 
 """

System_prompt = """
You are a random republican USA president or an old (before 2000) democrat president  (e.g. Bush, Trump, Bill Clinton).
Answer the following prompt while keeping a consistent style.
 
"""

def create_google_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    return client

def generate_prompts_for_speeches(speeches) -> list[str]:
    client = create_google_client()
    prompts = []
    for speech in tqdm(speeches):
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=generate_prompt.format(speech=speech)
        )
        prompts.append(response.text)
        sleep(120)
    return prompts

def generate_generic_responses(prompts):
    client = create_google_client()
    answers = []
    for prompt in prompts:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(system_instruction=System_prompt),
        )
        sleep(120)
        answers.append(response.text)
    return answers


if __name__ == "__main__":
    df_sample = pd.read_csv("cleaned_speeches.csv").sample(5)
    prompts = generate_prompts_for_speeches(df_sample["speech"])

    df_sample["prompt"] = prompts

    responses = generate_generic_responses(prompts)
    df_sample["wrong_speech"] = responses
    df_sample.to_csv("speech_prompts.csv", index=False, columns=["prompt", "speech", "wrong_speech"])

    df_turns = pd.read_csv("turns.csv").sample(5)
    questions = df_turns["question"]
    answers = generate_generic_responses(questions)
    df_turns["wrong_answer"] = answers
    df_turns.to_csv("turns_prompts.csv", index=False, columns=["question", "answer", "wrong_answer"])
