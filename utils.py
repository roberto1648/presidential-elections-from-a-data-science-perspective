import re
import pandas as pd


def make_dict_of_state_to_lowercase_and_no_space():
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
        'Colorado', 'Connecticut', 'Delaware', 'Dist. of Col.', 'Florida',
        'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'CD-1', 'CD-2', 'CD-3',
        'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming',
    ]
    new_names = map(lambda x: x.lower(), states)
    regex = re.compile('[^a-zA-Z]')
    new_names = map(lambda x: regex.sub('', x), new_names)

    return dict(zip(new_names, states))


def make_state_to_lower_dict():
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
        'Colorado', 'Connecticut', 'Delaware', 'Dist. of Col.', 'Florida',
        'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'CD-1', 'CD-2', 'CD-3',
        'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming',
    ]
    new_names = map(lambda x: x.lower(), states)
    regex = re.compile('[^a-zA-Z]')
    new_names = map(lambda x: regex.sub('', x), new_names)

    return dict(zip(states, new_names))


def translate_dictionary_to_full_statename(dictionary={}):
    full_name = make_dict_of_state_to_lowercase_and_no_space()
    new_dictionary = {}

    for name, value in dictionary.iteritems():
        if name in full_name:
            new_dictionary[full_name[name]] = value

    return new_dictionary


def get_state_neighbors(avoid=["AK", "DC"]):
    codes_df = pd.read_csv("data/us_states_and_code.csv")
    neigh_df = pd.read_csv("data/neighbors-states.csv")
    codes_dict = dict(zip(codes_df['Abbreviation'].values,
                          codes_df['State'].values))
    neigh_dict = {}

    for abbr in codes_df["Abbreviation"].values:
        if abbr not in avoid:
            neighbors = ["".join(row).replace(abbr, "") for row in neigh_df.values if abbr in row]
            neighbors = [codes_dict[x] for x in neighbors if x not in avoid]
            neigh_dict[codes_dict[abbr]] = neighbors

    return neigh_dict


def to_lower_letters(text=""):
    new_text = text.lower()
    regex = re.compile('[^a-zA-Z]')
    new_text = regex.sub('', new_text)
    return new_text