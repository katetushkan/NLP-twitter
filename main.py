import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub.utils import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

DATASET_URL = 'dataset/uk_pm.csv'


def sentiment_analysis_score_roberta(example, tokenizer, model):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


def print_hi(name):
    plt.style.use('ggplot')

    # Read In data from twitter dataset
    ds = pd.read_csv(DATASET_URL)
    ds = ds.head(10000)

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    pt_batch = tokenizer(
        ["U are very Bad!"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    output = model(**pt_batch)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    print(scores_dict)

    res = {}
    for i, row in tqdm(ds.iterrows(), total=len(ds)):
        try:
            text = row['text']
            myid = row['id']
            roberta_result = sentiment_analysis_score_roberta(text, tokenizer, model)
            res[myid] = roberta_result
        except RuntimeError:
            print(f'Broke for id {myid}')

    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'Id'})
    results_df = results_df.join(ds, how='left')

    print(results_df.columns)
    sns.jointplot(data=results_df, x='likecount', y='roberta_pos')
    sns.jointplot(data=results_df, x='likecount', y='roberta_neg')
    sns.jointplot(data=results_df, x='likecount', y='roberta_neu')
    plt.show()



if __name__ == '__main__':
    print_hi('PyCharm')

