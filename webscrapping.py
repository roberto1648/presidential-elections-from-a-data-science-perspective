import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import re


def main(year_from=1940, year_to=2016):
    for year in np.arange(year_from, year_to + 4, 4):
        url = "http://www.presidency.ucsb.edu/showelection.php?year={}"
        url = url.format(year)
        r = requests.get(url)
        html = r.text
        df = get_election_results_df(html)
        file_path = "data/presidential_elections/{}.csv"
        file_path = file_path.format(year)
        print year, list(df.columns[2::3])
        print
        df.to_csv(file_path, index=False)
        time.sleep(10 + 10 * np.random.rand())


def get_election_results_df(html=""):
    soup = BeautifulSoup(html, "lxml")
    # find the election results table
    tag = soup.find("table")
    # go to the table's column labels tag
    tag = soup.find(text=re.compile("STATE*"))
    # navigate to the table row tag
    tag = tag.parent.parent.parent
#     table_cols = [x.text.strip() for x in tag.find_all("td")]
    # get the candidate names or parti affiliation
    cand_tag = tag.previous_sibling.previous_sibling
    candidates = [x.text.strip() for x in cand_tag.find_all("td")]
    candidates = [x for x in candidates if len(x) > 0]
    
    rows = []

    # find the first state:
    tag = tag.next_sibling.next_sibling.next_sibling.next_sibling

    while True:
        if len(tag) > 1:
            row = [x.text.encode("utf-8") for x in tag.find_all("td")]
            row = np.array(row)
            if len(row[0]) > 0: rows.append(row)
            if "Totals" in row[0]: break

        tag = tag.next_sibling

        if not tag: break
    
    rows = np.array(rows)
    two_cols = ["STATE", "TOTAL VOTES"]
    df1 = pd.DataFrame(data=rows[:, :2], columns=two_cols)
    
    # make sure that the candidate number matches the number of data columns
    actual_ncandidates = (np.shape(rows)[1] - 2) // 3
    candidates = candidates[-actual_ncandidates:]
    # make a hierarchical index for each candidate
    nested_cols = [u'Votes', u'%', u'EV']
    iterables = [candidates, nested_cols]
    multi = pd.MultiIndex.from_product(iterables)#, names=['first', 'second'])
    df2 = pd.DataFrame(data=rows[:, 2:], columns=multi)
    
    df = pd.concat([df1, df2], axis=1)
    
    return df


if __name__ == "__main__":
    main()