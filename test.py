# import json
#
# idx_mapping = {
#     1: "A",
#     2: "B",
#     3: "C",
#     4: "D"
# }
#
#
# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s
#
#
# def convert_to_llm_prompt(question_data: str) -> str:
#     # Load the question data
#     data = json.loads(question_data)
#
#     # Build the LLM prompt
#     prompt = f"{data['question']}\n\n"
#
#     for idx, choice in enumerate(data['choices'], start=1):
#         prompt += f"{idx_mapping[idx]}. {choice}\n"
#
#     prompt += f"\nAnswer: {data['answer']}"
#
#     return prompt
#
#
# # Example usage
# question_data = """
# {
#   "question": "What is the embryological origin of the hyoid bone?",
#   "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
#   "answer": "D"
# }
# """
#
# llm_prompt = convert_to_llm_prompt(question_data)
# print(llm_prompt)

# data = {'question': ["Box a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?", 'One hundred years ago, your great-great-grandmother invested $100 at 5% yearly interest. What is the investment worth today?', "Krete is an unmarried taxpayer with income exclusively from wages. By December 31, year 1, Krete's employer has withheld $16,000 in federal income taxes and Krete has made no estimated tax payments. On April 15, year 2, Krete timely filed for an extension request to file her individual tax return, and paid $300 of additional taxes. Krete's year 1 tax liability was $16,500 when she timely filed her return on April 30, year 2, and paid the remaining tax liability balance. What amount would be subject to the penalty for underpayment of estimated taxes?", 'On January 1, year 1, Alpha Co. signed an annual maintenance agreement with a software provider for $15,000 and the maintenance period begins on March 1, year 2. Alpha also incurred $5,000 of costs on January 1, year 1, related to software modification requests that will increase the functionality of the software. Alpha depreciates and amortizes its computer and software assets over five years using the straight-line method. What amount is the total expense that Alpha should recognize related to the maintenance agreement and the software modifications for the year ended December 31, year 1?', 'An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?'], 'subject': ['accounting', 'professional_accounting', 'professional_accounting', 'professional_accounting', 'professional_accounting'], 'choices': [['$70,000', '$75,000', '$80,000', '100000'], ['$13,000', '$600', '$15,000', '$28,000'], ['$0', '$500', '$1,650', '$16,500'], ['$5,000', '$13,500', '$16,000', '$20,000'], ['Valuation and allocation', 'Completeness', 'Rights and obligations', 'Presentation and disclosure']], 'answer': [3, 0, 0, 1, 1]}
#
# index_list = [i for i, subject in enumerate(data['subject']) if subject == 'professional_accounting']
#
# new_data = [
#     {
#         'question': data['question'][i],
#         'subject': data['subject'][i],
#         'choices': data['choices'][i],
#         'answer': data['answer'][i]
#     }
#     for i in index_list
# ]
#
# for item in new_data:
#     print(item)

# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#
#
# def preprocess_function(example):
#     ending_names = ["ending0", "ending1", "ending2", "ending3"]
#     first_sentence = [example["sent1"]] * 4
#     print("FIRST_SENTENCE-->", first_sentence)
#     question_header = example["sent2"]
#     print("question_header-->", question_header)
#     second_sentence = [f"{question_header} {example[end]}" for end in ending_names]
#     print("second_sentence-->", second_sentence)
#     tokenized_examples = tokenizer(first_sentence, second_sentence, truncation=True)
#     print("tokenized_examples-->", tokenized_examples)
#     return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
#
#
# example = {
#     'ending0': 'passes by walking down the street playing their instruments.',
#     'ending1': 'has heard approaching them.',
#     'ending2': "arrives and they're outside dancing and asleep.",
#     'ending3': 'turns the lead singer watches the performance.',
#     'fold-ind': '3416',
#     'gold-source': 'gold',
#     'label': 0,
#     'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
#     'sent2': 'A drum line',
#     'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
#     'video-id': 'anetv_jkn6uvmqwh4'
# }
#
# # example2 = {
# #     'ending0': 'No he didnt.',
# #     'ending1': 'has heard approaching them.',
# #     'ending2': "Hello.",
# #     'ending3': 'hi.',
# #     'fold-ind': '3452435342',
# #     'gold-source': 'gold',
# #     'label': 2,
# #     'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
# #     'sent2': 'A drum line',
# #     'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
# #     'video-id': 'anetv_jkn6uvmqwh4'
# # }
#
# results = preprocess_function(example)
# # results2 = preprocess_function(example2)
#
# num_choices = len(results["input_ids"])
# flattened_features = [{k: v[i] for k, v in results.items()} for i in range(num_choices)]
#
# flattened_features = sum(flattened_features, [])
#
# print(flattened_features)

# import os
# import zipfile
#
# # Specify the parent directory and the extraction directory
# extraction_dir = r'C:\Users\gramsjoe\Desktop\DoD documents\STIG'
# parent_dir = r'C:\Users\gramsjoe\Downloads\U_SRG-STIG_Library'
#
# # Walk through all files in the parent directory
# for foldername, subfolders, filenames in os.walk(parent_dir):
#     for filename in filenames:
#         # Check if the file is a .zip file
#         if filename.endswith('.zip'):
#             # Create a full path to the .zip file
#             full_path = os.path.join(foldername, filename)
#
#             # Create a specific extraction path for the existed file
#             # (preserves the inner structures of the existed file)
#             specific_extraction_path = os.path.join(extraction_dir, os.path.splitext(filename)[0])
#
#             # Create the extraction directory if it doesn't exist
#             os.makedirs(specific_extraction_path, exist_ok=True)
#
#             # Open the .zip file and extract it to the specific extraction path
#             with zipfile.ZipFile(full_path, 'r') as zip_ref:
#                 zip_ref.extractall(specific_extraction_path)
#
#             print(f"Extracted all files from {full_path} to {specific_extraction_path}")
#
# print("Done unzipping all files!")

import os
import shutil

# Specify the parent directory and the target directory
parent_dir = r'C:\Users\gramsjoe\Desktop\STIG'
target_dir = r'C:\Users\gramsjoe\Desktop\DoD documents\STIG'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Walk through all files in the parent directory
for foldername, subfolders, filenames in os.walk(parent_dir):
    for filename in filenames:
        # Check if the file ends with "_Overview"
        if filename.endswith('_Overview.pdf'):
            # Create a full path to the file
            source = os.path.join(foldername, filename)
            destination = os.path.join(target_dir, filename)

            # Move the file to the target directory
            shutil.move(source, destination)

            print(f"Moved file {source} to {destination}")

print("Done moving all files!")