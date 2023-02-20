import re

import nltk
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# -----------------Variables and Data Structures --------------------
filename = "Code2.png"
search_list = []
all_the_lines = []
full_text = str()

# TODO: PLEASE POPULATE YOUR ENDPOINT URL AND KEY INI THE CREDENTIALS OR THE CODE WILL NOT WORK :-)
endpoint = ""
credential = AzureKeyCredential("")

# -------------------------------------
def gen_NER_tags(paramtext: str):
    tagged_text = pos_tag(word_tokenize(paramtext))
    # Look for all nouns and pronouns. These 'Things' tend to be Actors!
    for t in tagged_text:
        # Use a loop though you can use smaller lambda functions
        if t[1] == "NN" or t[1] == "NNS" or t[1] == "NNP" or t[1] == "NNPS" or t[1] == "PRP" or t[1] == "PRP$":
            search_list.append(t[0])
    #
    return search_list

# -------------------------------------
def most_important_pareto(paramtext: str):
    listed_text = word_tokenize(paramtext)
    extraction_metric = int(0.20 * round(len(listed_text), 0))
    # print(f"\n\nMost important phrases: {extraction_metric} of {len(listed_text)}")
    freqdist = nltk.FreqDist(samples=listed_text)
    top20percent = freqdist.most_common(extraction_metric)
    for t in top20percent: search_list.append(t[0])
    #
    return search_list
#
print("Connecting to Azure...")
azure_client = DocumentAnalysisClient(endpoint, credential)
#
print(f"Opening {filename}...")
with open(filename, "rb") as fd:
    document = fd.read()
#
print(f"Analysing {filename}...")
polled_outcome = azure_client.begin_analyze_document("prebuilt-layout", document)
#
print("Fetching results from Azure...")
outcome = polled_outcome.result()
# Loop through the resultset sent by powerful Form Recognizer endpoint
for page in outcome.pages:
    print(f"Source file {filename}, dimensions:{page.width} x {page.height}, metric unit: {page.unit}")
    for the_lineid, line in enumerate(page.lines):
        all_the_lines.append(line.content)
        full_text = full_text + line.content

# Clean punctuations unless you want these to be searchable as well :-)
to_clean = r',|\(|\)|\[|\]|;|:|!]|#|\.'
refined = re.sub(to_clean, '', full_text)
# Remove 1 character word that are not important unless you want them to be searachable as well :-)
to_clean = r' . '
refined = re.sub(to_clean, '', refined)
# Call functions to generate labels - this helps in search and as removes human from the loop
most_important_pareto(refined)
gen_NER_tags(refined)

# Make text unique to reduce overhead, redundancy
labels_or_search_strings = list(set(search_list))
schema_data = {"filename": filename, "labels": labels_or_search_strings}
# Remove the human in loop and label the image
df = pd.DataFrame(schema_data)
df.to_csv(filename + ".csv", index=False)
print(df)


