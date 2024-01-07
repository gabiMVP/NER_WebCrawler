import itertools
import json
import string
import re

import nltk
import numpy as np
import requests
import torchmetrics
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore

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

model_checkpoint = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

MAX_LEN = 512
SUMMARY_MAX_LEN=128
TRAIN_BATCH_SIZE=2
VALID_BATCH_SIZE=2
LEARNING_RATE = 3e-4
MAX_EPOCHS =10

def main(name):
    #download site Data only if not provided already
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
    #preprocess to Input Output form only if not already done or new data
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
                input, tag = getInputOutputNotTokenizedForSummary(current_xml, current_product_found)
                inputs.extend(input)
                tags.extend(tag)
        # save data to CVS
        trainDF = pd.DataFrame({'text': inputs, 'summary': tags})
        trainDF.to_csv('data100SitesNEW_Summary.csv', index=False, encoding='utf-8')


    # Data is ready here
    # trainDF = pd.read_csv('data100SitesNEW_trunc.csv', encoding='utf-8')
    trainDF = pd.read_csv('data100SitesNEW_Summary.csv', encoding='utf-8')
    trainDF.text = 'summarize: ' + trainDF.text
    train_size = 0.9
    train_dataset = trainDF.sample(frac=train_size)
    val_dataset = trainDF.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    class SummaryDataset(torch.utils.data.Dataset):

        def __init__(self, dataframe, tokenizer, source_len, summ_len):
            self.tokenizer = tokenizer
            self.data = dataframe
            self.source_len = source_len
            self.summ_len = summ_len
            self.summary = self.data.summary
            self.text = self.data.text

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            text = str(self.text[index])
            text = ' '.join(text.split())

            summary = str(self.summary[index])
            summary = ' '.join(summary.split())

            source = self.tokenizer.batch_encode_plus([text], max_length=self.source_len, pad_to_max_length=True,
                                                      return_tensors='pt')
            target = self.tokenizer.batch_encode_plus([summary], max_length=self.summ_len, pad_to_max_length=True,
                                                      return_tensors='pt')
            target['input_ids'][target['input_ids']==0] =-100
            source_ids = source['input_ids'].flatten()
            source_mask = source['attention_mask'].flatten()
            target_ids = target['input_ids'].flatten()

            return {
                'source_ids': source_ids.to(dtype=torch.long),
                'source_mask': source_mask.to(dtype=torch.long),
                'target_ids': target_ids.to(dtype=torch.long),
            }

    print(val_dataset.head())
    print(train_dataset.head())

    class SummaryDataModule(pl.LightningDataModule):
        def __init__(
                self,
                train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                tokenizer: T5Tokenizer,
                batch_size: int = 8,
                source_max_token_len: int = MAX_LEN,
                target_max_token_len: int = SUMMARY_MAX_LEN,
        ):
            super().__init__()
            self.train_df = train_df
            self.test_df = test_df
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.source_max_token_len = source_max_token_len
            self.target_max_token_len = target_max_token_len

        def setup(self):
            self.train_dataset = SummaryDataset(
                self.train_df,
                self.tokenizer,
                self.source_max_token_len,
                self.target_max_token_len
            )
            self.test_dataset = SummaryDataset(
                self.test_df,
                self.tokenizer,
                self.source_max_token_len,
                self.target_max_token_len
            )
            self.val_dataset = SummaryDataset(
                self.test_df,
                self.tokenizer,
                self.source_max_token_len,
                self.target_max_token_len
            )
        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4
            )

        def val_dataloader(self):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=4
            )

        def test_dataloader(self):
            return DataLoader(
                self.test_dataset,
                batch_size=1,
                num_workers=4
            )

    BATCH_SIZE =2
    N_EPOCHS = 2
    data_module = SummaryDataModule(train_dataset, val_dataset, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    class SummmaryModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)

        def forward(self, input_ids, attention_mask, labels=None):
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels)
            return output.loss, output.logits

        def training_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            loss, outputs = self(input_ids, attention_mask, labels)
            self.log("train_loss", loss, prog_bar=True, logger=True)
            return {"loss": loss, "predictions": outputs, "labels": labels}

        def validation_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            loss, outputs = self(input_ids, attention_mask, labels)
            self.log("val_loss", loss, prog_bar=True, logger=True)
            return loss

        def test_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            loss, outputs = self(input_ids, attention_mask, labels)
            self.log("test_loss", loss, prog_bar=True, logger=True)
            return loss

        def configure_optimizers(self):
            optimizer = AdamW(self.parameters(), lr=LEARNING_RATE)
            return optimizer

    model = SummmaryModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    logger = TensorBoardLogger("training-logs", name="Summary")
    trainer = pl.Trainer(
        # logger = logger,
        callbacks=[checkpoint_callback],
        max_epochs=N_EPOCHS,
        enable_progress_bar=True
    )


    trainer.fit(model, data_module)
    trainer.test()
    trained_model = SummmaryModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
    trained_model.freeze()  #

    def generate_summary(seed_line, model_, num_beam=4, penalty_length=0.2):

        # Put the model on eval mode
        model_.eval()
        encoded_query = tokenizer(seed_line,
                                  return_tensors='pt', pad_to_max_length=True, truncation=True, max_length=512)
        input_ids = encoded_query["input_ids"]
        attention_mask = encoded_query["attention_mask"]
        generated_answer = model.model.generate(input_ids, attention_mask=attention_mask,
                                                max_length=128, top_p=0.95, top_k=50)
        decoded_answer = tokenizer.decode(generated_answer[0])

        return decoded_answer

    # print out 10 cases for sanity check
    rouge = torchmetrics.text.rouge.ROUGEScore()
    test100 = val_dataset.head(10)
    test100.iloc(0)
    rougeL_fmeasure = []
    for text, summary in test100.values:
        summaryModel = generate_summary(text, model)
        print("----text----" + "\n" + text)
        print("----summary----" + "\n" + summary)
        print("----summaryModel----" + "\n" + summaryModel)
        result = rouge(summaryModel, summary)
        rougeL_fmeasure.append(result["rougeL_fmeasure"])

    arr = np.array(rougeL_fmeasure)
    print(len(arr))
    print(np.mean(arr))

    # get RougeL score  on Eval set to evaluate the model
    rouge = torchmetrics.text.rouge.ROUGEScore()
    test100 = val_dataset
    rougeL_fmeasure = []
    for text, summary in test100.values:
        summaryModel = generate_summary(text, model)
        result = rouge(summaryModel, summary)
        rougeL_fmeasure.append(result["rougeL_fmeasure"])

    arr = np.array(rougeL_fmeasure)
    print(len(arr))
    print(np.mean(arr))

    # test on a never before seen site
    pageTest = 'https://vauntdesign.com/'
    siteData = getTextAndProductsForSite(pageTest)

    paragraph = str(BeautifulSoup(siteData['string_page'][0][0], "xml"))
    paragraph = paragraph[500:1024]
    print("-------paragraph:-------" + "\n" + paragraph)

    summaryModel = generate_summary(paragraph, model)
    print("-------summary:-------" + "\n" + summaryModel)

def getInputOutputNotTokenizedForSummary(xml, list_product_per_xml):
    input = []
    tags = []
    # huge XML
    xml_string = xml[0]
    lines = xml_string.split('\n')
    last_index = 0
    product_len = len(list_product_per_xml) - 1
    textCumumated = ""
    for i, text in enumerate(lines):
        if last_index > product_len:
            break
        product = list_product_per_xml[last_index]
        # this BeautifulSoup is there just to handle HTML entities because these can happen and the texts won't match
        text1 = BeautifulSoup(text, "xml")
        if product in text or product in text1.getText():
            last_index += 1
            textCumumated += text
            input.append(textCumumated)
            tags.append(product)
            textCumumated = ""
        else:
            textCumumated +=  str(text)

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
    for mainpath in listpaths:
        r = requests.get(mainpath.text, headers=headers)
        responsemainPage = BeautifulSoup(r.content, "xml")
        x1 = responsemainPage.findAll("title")
        string_page = [str(responsemainPage)]
        product_page = [str(x.getText()) for x in x1]
        stringsPage.append(string_page)
        products_page.append(product_page)

    if len(stringsPage) == 0 or len(products_page) == 0:
        return None
    return {
        "string_page": stringsPage,
        "products_page": products_page,
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
