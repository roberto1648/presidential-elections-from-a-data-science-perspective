import numpy as np
import pandas as pd
import re
import utils


some_candidates = {
    "ROOSEVELT": "democrat",
    "WILLKIE": "republican",
    "DEWEY": "republican",
    "TRUMAN": "democrat",
    "EISENHOWER": "republican",
    "STEVENSON": "democrat",
    "JOHNSON": "democrat",
    "GOLDWATER": "republican",
    "NIXON": "republican",
    "McGOVERN": "democrat",
    "REAGAN": "republican",
    "MONDALE": "democrat",
    "BUSH": "republican",
    "DUKAKIS": "democrat",  
}


def main(year_from=1940, year_to=2016):
    diff_list = []

    for year in np.arange(year_from, year_to + 4, 4):
        file_name = "data/presidential_elections/{}.csv".format(year)
        df = pd.read_csv(file_name)
        year_diff = calculate_dem_rep_state_differences(df)
        year_diff.index = [year]
        diff_list.append(year_diff)

    diff_df = pd.concat(diff_list, sort=True)
    rearranged_cols = [x for x in diff_df.columns if "total" not in x] + ["total"]
    diff_df = diff_df[rearranged_cols]
    diff_df.to_csv("data/dem-rep_diff_per_state.csv")

    return diff_df


def find_percentage_column_names(df_columns=[(),]):
    perc_cols = [x for x in df_columns if "%" in x]
    
    if not perc_cols:
        perc_cols = [x for x in df_columns if "%" in x[1]]
        
    return perc_cols


def find_democrat_republican_column_names(columns=[(),]):
    democrat_col = [x for x in columns if ("democrat" in "".join(x).lower())]
    
    if democrat_col:
        democrat_col = democrat_col[0]
    else:
        for x in columns:
            for name, value in some_candidates.iteritems():
                if ((name in x[0]) or (name in x)) and value == 'democrat':
                    democrat_col = x
                    break
    
    republican_col = [x for x in columns if ("republican" in "".join(x).lower())]
    
    if republican_col:
        republican_col = republican_col[0]
    else:
        for x in columns:
            for name, value in some_candidates.iteritems():
                if ((name in x[0]) or (name in x)) and value == 'republican':
                    republican_col = x
                    break
                
    return democrat_col, republican_col


def find_states_column_name(df_columns=[]):
    state_col = ""
    
    for name in df_columns:
        if re.match(re.compile("STATE*") , name):
            state_col = name
            break
        
    assert state_col != ""
    
    return state_col


def get_states_row_indices(df=pd.DataFrame()):
    state_col_name = find_states_column_name(df.columns)
    s = df[state_col_name]
    indices = []
    
    for index, value in s.iteritems():
        cond1 = "cd-" not in value.lower()
        cond2 = not re.match(re.compile("total*") , value.lower())
        cond3 = len(value.strip()) > 0
        
        if cond1 and cond2 and cond3: indices.append(index)
            
    return list(set(indices))


def get_totals_row_index(df=pd.DataFrame()):
    state_col_name = find_states_column_name(df.columns)
    s = df[state_col_name]
    found = False
    
    for index, value in s.iteritems():
        if re.match(re.compile("total*") , value.lower()): 
            found = True
            break
    
    assert found == True
    
    return index


def process_percentage_col_element(x):
    if type(x) is str:
        if x.strip():
            xout = x.replace("%", "").strip()
            xout = float(xout)
        else:
            xout = np.nan
    elif type(x) is float:
        xout = x
    else:
        raise "not expected x type: {}".format(type(x))
    
    return xout


def get_processed_states_column(df):
    state_indices = get_states_row_indices(df)
    
    states_col_name = find_states_column_name(df.columns)
    states_col = df[states_col_name]
    states_col = states_col.loc[state_indices]
    states_col = states_col.map(lambda x: x.lower())
    regex = re.compile('[^a-zA-Z]')
    states_col = states_col.map(lambda x: regex.sub('', x))
    
    return states_col


def calculate_dem_rep_state_differences(df=pd.DataFrame()):
    perc_col_names = find_percentage_column_names(df.columns)
    dem_col_name, rep_col_name = find_democrat_republican_column_names(perc_col_names)
    dem_col = df[dem_col_name].map(process_percentage_col_element)
    rep_col = df[rep_col_name].map(process_percentage_col_element)
    
    totals_index = get_totals_row_index(df)    
    state_indices = get_states_row_indices(df)
    indices = state_indices + [totals_index]
    
    states_col = get_processed_states_column(df)
    
    data = dem_col.loc[indices] - rep_col.loc[indices]
    data = data.values
    name_dict = utils.make_dict_of_state_to_lowercase_and_no_space()
    index = states_col.loc[state_indices].values
    index = [name_dict[x] for x in index]
    index += ["total"]
    diff = pd.DataFrame(data=[data], columns=index)
    
    return diff


if __name__ == "__main__":
    main()

