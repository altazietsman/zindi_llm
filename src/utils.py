from rouge_score import rouge_scorer

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def calc_rouge_score(pred, actual):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    precision, recall, fmeasure = scorer.score(actual, pred)['rouge1']

    return fmeasure
