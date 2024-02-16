import pandas as pd
import os
import re
import pathlib


def read_booklets(dir_path: str) -> pd.DataFrame:
    """Reads all excel sheets of the booklets

    Arguments:
    ----------
    dir_path: str
              path of where booklets are stored

    Return:
    -------
    df_booklet: pandas dataframe
                dataframe oof booklet text
    """

    files = os.listdir(pathlib.Path(dir_path))

    booklets = []

    for file in files:
        if file.startswith("book"):
            df_booklet = pd.read_excel(dir_path + file)
            df_booklet["book"] = file[:8]
            df_booklet.columns = ["index", "text", "book"]
            booklets.append(df_booklet)

    df_booklet = pd.concat(booklets)
    df_booklet["text"] = df_booklet["text"].astype("str")

    # print(df_booklet.head())
    # print(df_booklet[(df_booklet['index'] == 140) & (df_booklet['book'] == 'booklet6')])
    return df_booklet


def clean_text(text):
    """Cleans a given DataFrame row's text.

    Arguments:
    ----------
    text: str
              booklet text

    Return:
    -------
    text: str
                cleaned text
    """
    # Remove newline characters
    text = text.replace("\n", "")

    # Remove non-alphabetic characters and extra spaces
    text = re.sub(r"[^A-Za-z0-9 /.,!?]+", "", text)


    # Remove specific substring 'BOOKLET xxx'
    text = text.replace("BOOKLET ONE", "")
    text = text.replace("BOOKLET TWO", "")
    text = text.replace("BOOKLET THREE", "")
    text = text.replace("BOOKLET FOUR", "")
    text = text.replace("BOOKLET FIVE", "")
    text = text.replace("BOOKLET SIX", "")

    # Replace spaces that are incorrectly encoded
    text = text.replace(u'\xa0', u' ')

    # Convert duplicated spaces to single spaces
    text = re.sub(r'\s+', ' ', text)

    # Check if the cleaned text is empty or contains only spaces
    if text.isupper():
        text = ""  # Return None for rows to be removed

    return text


def preprocess_booklets(booklet_dir="./data/data/booklets/", save_to_csv=True):
    df_booklet = read_booklets(dir_path=booklet_dir)
    df_booklet["cleanText"] = df_booklet["text"].apply(clean_text)
    # Remove empty rows from dataframe
    df_booklet = df_booklet[df_booklet["cleanText"].str.strip() != ""]

    # Remove rows where cleanText length is less than 15 chars
    df_booklet = df_booklet[(df_booklet["cleanText"].str.len() >= 15)]

    # print(df_booklet[(df_booklet['index'] == 140) & (df_booklet['book'] == 'booklet6')])

    if save_to_csv:
        df_booklet.to_csv("./data/data/resources/booklet_clean.csv")

    return df_booklet
