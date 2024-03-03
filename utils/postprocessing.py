import re

def first_letter(s):
    """Returns the index of the first letter of each keyword
    
    Argument:
    ---------
    s: str
    """
    
    m = re.search(r'[a-z]', s, re.I)
    if m is not None:
        return m.start()
    return -1

def format_keywords(keywords:str):
    """Formats key words
    
    Argument:
    ---------
    keywords: str
              string of keywords used
    
    Returns:
    parsed_keywords: str
                     string of keywords list
    """
    keywords_list = keywords.split("Keywords:")[-1].split(',')
    if len(keywords_list) == 1:
        keywords_list = keywords.split("Keywords:")[-1].split('\n')
    keywords_indexes = [first_letter(word) for word in keywords_list]
    clean_keywords = [keywords_list[i][keywords_indexes[i]:] for i in range(len(keywords_list))]
    parsed_keywords = ", ".join(sorted(list(set([i.lstrip().capitalize().replace('\n', "") for i in clean_keywords]))))
    return parsed_keywords.title()

def clean_book_info(book_info: dict):
    """Reformats book and paragraph info
    
    arguments:
    ----------
    book_info: dict
               dictionary with book numbers and paragrahs
    Returns:
    --------
    book_info: dict
    """
    
    if len(set(book_info['book'].split(","))) > 1:
        book_info['book'] = book_info['book'].split(",")[0]
        book_info['paragraph'] = book_info['paragraph'].split(",")[0]

    else: 
        book_info['book'] = book_info['book'].split(",")[0]
    return book_info