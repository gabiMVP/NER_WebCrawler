# NER with dataset created from scraping the web

The purpose is to do Name Entity Recognition where there is only one entity:Product 

Here I did fine-tuning using the standard distilbert model and also trying with xlm-roberta-base
These pretrained models are taken from the Huggingface 

### Dataset Aquisition
Our list of sites from where we get the products are Web shops.Web shops do a lot of SEO to be seen 
by search engines and actually facilitate search engines scraping them.For this they have robots.txt file
which is there to tell the scrapping program where it can find useful information the site wants to 
be broadcasted to the world

I use root site path  + /robots.txt and see what links are there , the most important link is the sitepath

I get the sitepath link from /robots.txt then navigate all link from it 

the content from those links will be xml documents where we see that  title tag   corresponds to products 

There is a link above the title tag upper in the XML tree , the  loc tag where if we access it we see 
the individual product page , this would be more useful if we want to get other data like price.
We would in that case navigate to that link and scrape from there 

I  downloaded the XML of each page and the list of product extracted from <loc> tag


### Dataset preprocessing 

So far the entry data  is a large XML and the output data  is a list of product names  

The XML is split into lines and for each line into words.
For each of the lines we know if it's the line where we have products from the product names list.
We then associate for each word in the line a NER tag.


The final dataset is in the form of a CVS file which has the structure:
List of words of words of line | List of NER tags associated for those words of line

<img src="./dataframe.png" alt="Confusion Matrix" />

### Implementation notes:

I did the same implementation as presented in this link:

https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5

I tried fine tunning 2 pre-trained models
- "distilbert-base-uncased"
- "xlm-roberta-base" 

It is important to note that the seqeval library from HuggingFace does not work if the data is not
in the standard NER structure 
that is why I used the standards suffixes for the labels 
- B-PRODUCT    where B prefix as Beginning word of product 
- I-PRODUCT    where I prefix as Mid word of product 
- E-PRODUCT    where E prefix as End word of product 

The models were fine tunned only on 244008 input output pairs corresponding to the data in data10Sites.csv

### Results Bert:

| Epoch | Training_Loss | Validation_Loss | Precision | Recall    | F1_score  |Accuracy|
|-------|---------------|-----------------|-----------|-----------|-----------|-----------|
| 1     | 	0.000500     | 	0.007317       | 	0.963814 | 	0.942717 | 	0.953149 |	0.996491 |
| 2     | 	0.000200     | 	0.013572       | 	0.982870 | 	0.988706 | 	0.985780 |	0.998216 |
| 3     | 	0.000100     | 	0.013838       | 	0.980622 | 	0.983933 | 	0.982275 |	0.998172 |


### Notes
I used Google Colab to train the models , the Jupyter Notebooks are in the project files

