import re
import pandas as pd
def get_speakers(speeches):
    speakers = re.findall("(?:Mr|Ms|Mrs|Dr)?\.?\s?[A-Z][A-Za-z]{3,16}\s?:", speeches)
    return set(speakers)

def segment_speeches(text):
    speeches = re.split("\n\s+(?:delivered|Delivered)", text)
    return speeches
def get_parties():

    second_parties = {'Quotation Source', "Congressman Hensarling", "President-Elect Trump", "Q", "Congressman Ryan", "Congressman Pence", "Mr. Melhem", 'Que stion', 'Blackburn', 'Question', 'MINISTER NETANYAHU', 'Luther', 'Burundians', 'Audience', 'HAPLAIN RUTHERFORD', 'Sebelius', 'MILITARY AIDE', 'Audience Member', 'Au dience Member', 'QUESTION'}
    second_parties_2 = {'Dr. Benjamin', ' Prague', ' Kansas', ' Mason', ' Castro',  ' Cubs', ' Blackburn', ' Court', ' Everybody', ' King', ' Poland', ' Gandhi', 'Statement',  'Congress', 'Sebelius', 'Ms. Kahumbu', 'Francisco', 'Ms. Shumate', ' Question', 'Ms. Kliff', ' Sebelius', ' Audience', ' Side', ' America', ' Merkel', ' Thompson', 'Lincoln', 'Clinton', 'Hamilton', 'Aide', 'Illinoisans', 'Mr. Odede', 'Medvedev', 'Child', 'Mr. Klein', ' Shimon', 'Beginning', 'Facebook', ' Members', 'Forces', 'Depression', 'Cabinet', 'Ilves', 'Smith', 'Sasha:', ' Botticelli', 'Duncan', 'Bent', 'Pham',  'Secretary', 'Luther', 'Moderator', 'Timothy', 'Mr. Carney', 'Christopher', 'Macklemore', 'Arizona', 'Constitution', 'Isaiah', 'Question', 'Romney', 'Hensarling', 'Coast', ' Chafetz:', ' Ryan:', ' Academy:', ' States:', 'Mr. Earnest:', ' Required:', 'Connell:', 'Clinton:', ' Cregan:', ' Joel:', ' Strauss:', ' First:', 'Ms. Momposhi:', ' Leno:', ' Morehouse:', ' Member:', ' Dick:', ' Eagles:', ' Clay:', ' Nado:', ' Hang:', ' Luther :', ' Biden:', ' Matthew:', ' Second:', ' Auditor:', ' Point:', 'Mr. Melhem:', ' Roskam:', ' John:', ' Burwell:', 'Ms. Crowley:', ' Everett:', ' Israel:', ' Brooks:', ' Moore:', 'Romney:', ' Johnson:', ' Giese:', ' Harvey:', 'Mr. Lalampaa:', 'Ms. Dixon:', ' Americans:', ' Five:', ' Scripture:', ' Commons:', ' Initiative:', ' Pence:', ' Band:', ' Audience :', ' Member :', ' Comment:', ' Burundians :', ' Aeschylus:', ' Source :', ' Elliott:', 'Dr. Brumage:', ' Ezekiel:', 'Mr. Schieffer:', ' Aquino:', ' Trump:', ' Hollande:', ' Parry:'}
    second_parties_2 = second_parties.union(second_parties_2)
    second_parties = {party.replace(":","").strip() for party in second_parties_2}
    return second_parties_2

all_speeches = []
questions = []
answers = []
cleaned_speeches = []

def clean_speech(speech):
    cleaned_text = re.sub(r'^.*?\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4},?.*?\n', '', speech).strip()
    return cleaned_text

with open("all_obama_speeches.txt") as file:
    with open("all_obama_speeches_clean.txt", "a") as quotes_to_write_file:

        text = file.read()
        speeches = segment_speeches(text)
        for speech in speeches:
            question_answer_found=False
            for speaker in get_speakers(speech):
                start = 0
                if speaker.strip().replace(":","").strip() in get_parties():
                    turns = re.findall(f"{speaker}[\s\S]+?(?:President obama|President Obama|obama|Obama|President|OBAMA)\s?:[\s\S]+?(?={speaker}|\n{4:10})", speech)
                    for turn in turns:
                        question_answer = re.split("(?:President obama|President Obama|obama|Obama|President|OBAMA)\s?:", turn)
                        question = question_answer[0]
                        answer = question_answer[1]
                        question = re.sub(f"^.*?{speaker}","",question)
                        if question.strip() and answer.strip():
                            question = re.sub("[\s\n\r]+", " ", question)
                            questions.append(question)
                            answer = re.sub("[\s\n\r]+"," ", answer)
                            answers.append(answer.strip())
                            question_answer_found=True
            if question_answer_found: ## Remove turns to see if there is a speech left
                for speaker in get_speakers(speech):
                    start = 0
                    if speaker.strip().replace(":","").strip() in get_parties():
                        speech = re.sub(f"{speaker}[\s\S]+?(?:President obama|President Obama|obama|Obama|President|OBAMA)\s?:[\s\S]+?(?={speaker}|\n{4:10})", "", speech)


                if len(cleaned_speech) > 20:
                    cleaned_speech = clean_speech(speech)
                    cleaned_speech = re.sub("[\s\n\r]+", " ", cleaned_speech)
                    cleaned_speeches.append(cleaned_speech)

            else:

                cleaned_speech = clean_speech(speech)
                cleaned_speech = re.sub("[\s\n\r]+", " ",cleaned_speech)
                cleaned_speeches.append(cleaned_speech)

            quotes_to_write_file.write(speech)

df = pd.DataFrame({"question":questions, "answer":answers})
df.to_csv("turns.csv", index=False)

df = pd.DataFrame({"speech": cleaned_speeches})
df.to_csv("cleaned_speeches.csv" , index=False)