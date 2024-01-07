import itertools
import json
import string
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
from ast import literal_eval
import seqeval

"""
Since our target sites use SEO we can most likely infer they have robots.txt or sitemaps
the best way to get the sitemap is from Robots.txt where there will be a link to it 
in the sitemap will be more links but some for sure will have all the product list, webshops do this for SEO 
so our webscraper actually is taking the products from the XML of the pages linked in the sitemap
"""

ROBOTS = '/robots.txt'
Sitemaps = [
    '/sitemap.xml',
    '/sitemap-index.xml'
    '/sitemap.php'
    '/sitemap.txt'
    '/sitemap.xml.gz'
    '/sitemap/'
    '/sitemap/sitemap.xml'
    '/sitemapindex.xml'
    '/sitemap/index.xml'
    '/sitemap1.xml'
]
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
}

label_list = ['O', 'I-PRODUCT', 'B-PRODUCT', 'E-PRODUCT']
# always use the max label <nr of classes otherwise error at CrossEntropy
label_encoding_dict = {'O': 0, 'I-PRODUCT': 1, 'B-PRODUCT': 2, 'E-PRODUCT': 3}

task = "ner"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16


def main(name):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # download site Data only if not provided already
    data_already_downloaded = True
    if not data_already_downloaded:
        df = pd.read_csv('./data/furniture stores pages.csv')
        df = df.dropna()
        df_base_URL = df['max(page)'].map(getBasepage)
        df_base_URL_np = df_base_URL.values
        datasetList = []
        # download site data , XML and list of product for each site and put in list
        for i in range(100):
            url = df_base_URL_np[i]
            try:
                siteData = getTextAndProductsForSite(url)
                datasetList.append(siteData)
            except:
                continue
        # remove None elements which correspond to sites that give connection exception
        datasetListCleanNone = [i for i in datasetList if i is not None]
        # save data as JSON
        with open("datasetRaw1.json", "w") as outfile:
            json.dump(datasetListCleanNone, outfile)

    # preprocess to Input Output form only if not already done or new data
    data_already_preprocessed = True
    if not data_already_preprocessed:
        with open('datasetRaw1.json') as user_file:
            parsed_json = json.load(user_file)
        inputs, tags = [], []
        # from the downloaded data pre process in Input Output form

        for siteDict in parsed_json:
            list_xmls = siteDict['string_page']
            list_product_per_xml = siteDict['products_page']
            lenght = len(list_xmls)

            for i in range(lenght):
                current_xml = list_xmls[i]
                current_product_found = list_product_per_xml[i]
                input, tag = getInputOutputNotTokenized(current_xml, current_product_found)
                inputs.extend(input)
                tags.extend(tag)
        # save data to CVS
        trainDF = pd.DataFrame({'text': inputs, 'summary': tags})
        trainDF.to_csv('data100SitesNEW_trunc.csv', index=False, encoding='utf-8')

    # Data is ready here
    trainDF = pd.read_csv('data100SitesNEW_trunc.csv', encoding='utf-8')
    # we use literal_eval because when the CVS is saved list of strings becomes String in each entry
    trainDF['tokens'] = trainDF['tokens'].apply(literal_eval)
    trainDF['ner_tags'] = trainDF['ner_tags'].apply(literal_eval)

    dfSize = trainDF.shape[0]
    trainSize = dfSize * 0.90
    trainSize = int(trainSize)

    df_train = trainDF.iloc[:trainSize, :]
    df_test = trainDF.iloc[trainSize:, :]
    assert (df_test.shape[0] + df_train.shape[0] == dfSize)
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)
    id2label = {
        0: "O",
        1: "I-PRODUCT",
        2: 'B-PRODUCT',
        3: 'E-PRODUCT'
    }
    label2id = {
        "O": 0,
        "I-PRODUCT": 1,
        'B-PRODUCT': 2,
        'E-PRODUCT': 3
    }
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list),
                                                            id2label=id2label, label2id=label2id)

    args = TrainingArguments(
        f"test-{task}",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=1e-5,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                            zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                       zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"],
                "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model('variant.model')

    # Sanity check on some example from dataset
    pathSaved = '/content/DistilBertNER.model'
    tokenizer = AutoTokenizer.from_pretrained(pathSaved)
    sample = df_test.iloc[19]
    sample_text = sample['tokens']
    sample_tags = sample['ner_tags']

    sample_text = str(sample_text)
    print("sample_text" + sample_text)
    tokens = tokenizer(sample_text)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    model = AutoModelForTokenClassification.from_pretrained(pathSaved, num_labels=len(label_list))
    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                                attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    print("pred :" + str(predictions))
    predictions = [label_list[i] for i in predictions]
    print("pred :" + str(predictions))

    words = tokenizer.batch_decode(tokens['input_ids'])
    x = pd.DataFrame({'ner': predictions, 'words': words}).to_csv('BERT.csv')
    x = pd.DataFrame({'ner': predictions, 'words': words})
    print(x)

    # Sanity check on a never before seen site
    pageTest = 'https://vauntdesign.com/'
    siteData = getTextAndProductsForSite(pageTest)
    print(siteData['string_page'][0])
    print(siteData['products_page'][0])

    tokenizer = AutoTokenizer.from_pretrained(pathSaved)

    paragraph = str(BeautifulSoup(siteData['string_page'][0][0], "xml"))
    paragraph = paragraph[500:1024]
    print("paragraph:" + paragraph)
    list_lines = paragraph.split("\n")
    tokens = tokenizer.batch_encode_plus(list_lines)
    dataFrame = pd.DataFrame()
    print(tokens)
    model = AutoModelForTokenClassification.from_pretrained(pathSaved, num_labels=len(label_list))
    # since in training the model saw only line per line of the xml we have to preprocess and give it the input in the same format
    # with full paragraph the results are not as good
    for i, line in enumerate(tokens['input_ids']):
        print(i)
        print(line)
        predictions = model.forward(input_ids=torch.tensor(tokens['input_ids'][i]).unsqueeze(0),
                                    attention_mask=torch.tensor(tokens['attention_mask'][i]).unsqueeze(0))
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        predictions = [label_list[i] for i in predictions]
        words = tokenizer.batch_decode(tokens['input_ids'][i])
        df1 = pd.DataFrame({'ner': predictions, 'words': words})
        print(df1)
        dataFrame = pd.concat([dataFrame, df1], axis=0)
    x = dataFrame.to_csv('BERT.csv')


#  same code as huggingface  https://huggingface.co/docs/transformers/tasks/token_classification
def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def getInputOutputNotTokenized(xml, list_product_per_xml):
    input = []
    tags = []
    xml_string = xml[0]
    lines = xml_string.split('\n')
    # string_no_XML_tags  = BeautifulSoup(xml_string, "xml").getText()
    # lines1 = string_no_XML_tags.split('\n')
    last_index = 0
    product_len = len(list_product_per_xml) - 1
    for i, text in enumerate(lines):
        if last_index > product_len:
            break
        product = list_product_per_xml[last_index]
        # this BeautifulSoup is there just to handle HTML entities because these can happen and the texts won't match
        text1 = BeautifulSoup(text, "xml")
        # we add a space before starting and closing tag so text split " " can split without removing the <>
        text = text.replace("<", " <")
        text = text.replace(">", "> ")
        textList = text.split(" ")
        # remove empty space that be create by adding space
        textList = [N for N in textList if N != ""]
        if product in text or product in text1.getText():
            last_index += 1
            textToAdd = text1.getText().split(" ")
            # input.append(textToAdd)
            idx_start = text.find(textToAdd[0])
            idx_end = text.rfind(textToAdd[-1]) + len(textToAdd[-1])
            string_before_target = "".join(text[0:idx_start])
            string_after_target = "".join(text[idx_end:])
            list_string_before_target = string_before_target.split(' ')
            list_string_after_target = string_after_target.split(' ')
            list_string_before_target = [N for N in list_string_before_target if N != ""]
            list_string_after_target = [N for N in list_string_after_target if N != ""]
            tags_beggining = ['O'] * len(list_string_before_target)
            tags_end = ['O'] * len(list_string_after_target)
            input.append(list_string_before_target + textToAdd + list_string_after_target)
            # tags.append(['PRODUCT'] * len(textToAdd))
            listProduct = ['I-PRODUCT'] * len(textToAdd)
            listProduct[0] = 'B-PRODUCT'
            listProduct[-1] = 'E-PRODUCT'
            # tags.append(listProduct)
            tags.append(tags_beggining + listProduct + tags_end)
        else:
            input.append(textList)
            tags.append(['O'] * len(textList))
    return input, tags


def getTextAndProductsForSite(url):
    try:
        requests.get(url, headers=headers)
    except:
        print("site" + url + "is down")
        return
    urlRobots = url + ROBOTS
    r = requests.get(urlRobots, headers=headers)
    m2 = BeautifulSoup(r.content, "html.parser")
    listProp = m2.text.split('\n')
    sitemap = extractSiteMapfromRobotsTxt(listProp)
    if '' == sitemap:
        sitemap = tryKnownSitepath(url)
        if '' == sitemap:
            print("could not get data for site :" + url)
            # break for loop since we dont know the sitepath
            return

    r1 = requests.get(sitemap, headers=headers)
    sitemapXml = BeautifulSoup(r1.content, "xml", from_encoding='utf-8')
    # in sitemap we have links we go in one by one,  some of these will contain the products
    listpaths = sitemapXml.findAll('loc')
    stringsPage = []
    products_page = []
    dictList = []
    for mainpath in listpaths:
        r = requests.get(mainpath.text, headers=headers)
        responsemainPage = BeautifulSoup(r.content, "xml")
        x1 = responsemainPage.findAll("title")
        # string_page = [responsemainPage.getText()]
        string_page = [str(responsemainPage)]
        product_page = [str(x.getText()) for x in x1]
        stringsPage.append(string_page)
        products_page.append(product_page)
        # dict ={
        #     "string_pageX": string_page,
        #     "products_pageX": product_page
        # }
        # dictList.append(dict)

    if len(stringsPage) == 0 or len(products_page) == 0:
        return None
    return {
        "string_page": stringsPage,
        "products_page": products_page,
        # "dict":dictList
    }


def extractSiteMapfromRobotsTxt(listProp):
    sitemap = ''
    for line in listProp:
        try:
            l1 = line.split(' ')
            if l1[0] == 'Sitemap:':
                sitemap = l1[1]
        except:
            pass
    if sitemap == "":
        # use sitemap list
        pass
    return sitemap


def getBasepage(link):
    parsed = urlparse(link)
    base = parsed.netloc
    scheme = parsed.scheme
    page = scheme + '://' + base
    return page


def tryKnownSitepath(url):
    for sitePath in Sitemaps:
        urltried = url + sitePath
        r1 = requests.get(urltried)
        if (r1.status_code == 200):
            return urltried
    return ""


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
