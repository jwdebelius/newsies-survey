from textwrap import TextWrapper

import numpy as np
import pandas as pd

# ### Wrappers
note_wrapper = TextWrapper(width=80,
                           initial_indent = '   * ', 
                           subsequent_indent='     ',
                           break_long_words=False)
check_wrapper = TextWrapper(width=55, 
                            initial_indent='  ', 
                            subsequent_indent='           ', 
                            break_long_words=False)
question_wraper = TextWrapper(width=78, 
                              break_long_words=False)
rank_wrapper = TextWrapper(width=45, 
                            initial_indent='  ', 
                            subsequent_indent='           ', 
                            break_long_words=False)


### Question type summary
def summarize_checks(prefix_, data_dict, data, spacer=55, rename=True):
    """
    Produces a count summary for categorical columns
    """
    cols = [c for c in data_dict.index if (prefix_ in c) & ('_[' in c)]
    original_answers = data_dict.loc[cols, 'original_answer']
    
    counts = pd.concat(axis=1, objs=[
        data[col_].value_counts(dropna=False) for col_ in cols
    ]).T
    counts.sort_index(axis='columns', inplace=True)
    counts.rename(
        columns={0: 'No', 1: 'Yes', 
                 '0.0': 'No', '1.0': 'Yes', 
                 '0': 'No', '1': 'Yes'}, 
        inplace=True)
    if rename:
        counts2 = counts.copy().rename(original_answers)
    else:
        counts2 = counts.copy()

    table_text = _build_bool_table(counts2.copy())

    return cols, original_answers, counts, table_text


def summarize_free_text(prefix_,  data_dict, data, spacer=55, rename=True):
    """
    Makes a pretty multiple response question
    """
    original = data_dict.loc[prefix_, 'original']
    counts = (data[prefix_] != 'chose not to respond') * 1
    
    text_table = [
        '     This is a free response question. Answers are not made public '
        'to protect',
        '     participant privacy.',
        '          ----------------------------------------------------------'
        '-      ',
        '     {0:>3d} participants chose to respond to this question'.format(
            counts.sum()),
    ]
    return original, counts, '\n'.join(text_table)


def summarize_multiple_choice(prefix_, data_dict, data, spacer=40, 
                              rename=True):
    """
    Turns multiple choice questions into a summary table
    """
    
    col_ = [c for c in data_dict.index if prefix_ in c][0]
    original = data_dict.loc[col_, 'original']
    order_ = data_dict.loc[col_, 'options']
    if pd.isnull(order_):
        order_ = np.sort(data[col_].dropna().unique())
    else:
        order_ = order_.split(" @ ")

    counts = data[col_].value_counts()[order_]
    
    text_table = _build_radio_table(counts.copy())
    
    return col_, counts, order_, text_table


def summarize_ranks(prefix_, data_dict, data, spacer=55, rename=True):
    """
    Produces a summary for ranked data
    """
    cols = [c for c in data_dict.index if prefix_ in c]
    original_answers = data_dict.loc[cols, 'original_answer']
    
    # Determines how many samples were missing and removes them
    missing = (original_answers[cols[0]] == 'chose not to respond')
    ranks = data[cols].copy().replace({'chose not to respond': np.nan})
    ranks.dropna(axis=0, how='all', inplace=True)
    
    # Gets the number of non-zero ranks
    ranks.replace({'7.0': np.nan}, inplace=True)
    describe = ranks.astype(float).describe().T
    if rename:
        describe.rename(original_answers, inplace=True)
    
    text_table = _build_rank_table(describe, missing)
    
    return cols, original_answers, describe, text_table


# ### Text clean up functions
def print_summary(col_, data_dict, data, counts=None, show_notes=True,
                  spacer=None):
    """
    Prints a summary
    """
    if spacer is None:
        spacer = '{0}'.format(''.join(['='] * 80))
    # Gets the question information
    short_name = col_.split('_[')[0]
    item = data_dict.loc[col_, 'item'].split('.')[0]
    question, type_, free_text = \
        data_dict.loc[col_, ['original', 'question_type', 'any_free_text']]
    question = '\n'.join(question_wraper.wrap(question))
    
    string = [
        f'Item {item}: {short_name} ({type_})',
        spacer,
        f'\n{question}\n',
        spacer,
    ]
    
    if counts is not None:
        string.append(f'\n{counts}\n\n{spacer}')
    
    # Gets the notes
    prefix = col_.split('_[')[0]
    cols = [c for c in data_dict.index if prefix in c]
    notes = data_dict.loc[cols, 'notes'].dropna()
    if (len(notes) > 0) & show_notes:
        notes = _format_notes(notes)
        string.append('Data cleaning notes\n-------------------')
        string.append(notes)
 
    string = '\n'.join(string)
    
    return string

    
def _format_notes(notes):
    """
    Makes the notes pretty
    """
    notes = ' @ '.join(np.hstack(notes)).split(" @ ")
    wrapped_notes = np.hstack([note_wrapper.wrap(note) for note in notes])
    
    return '\n'.join(wrapped_notes)


def _count_missing(table=None, num_=None):
    if (num_ is None) and (table is None):
        raise ValueError('A table or number must be supplied')
    if (num_ is None) and ('chose not to respond' not in table.index):
        num_ = 0
    elif (num_ is None):
        num_ = table['chose not to respond']
    
    if num_ == 0:
        return None

    participants = {1: 'participant'}.get(num_, 'participants')

    return f'{num_:>3} {participants} chose not to respond'


def _build_bool_table(counts2, spacer=55):
    missing_ = _count_missing(counts2.iloc[0])
    table = [
        ''.join(['     ', ''.join(['-'] * (spacer + 15))]),
        ''.join(['  Answer'.ljust(spacer), '  Yes', '    No']),
        ''.join(['-'] * (spacer + 15)),
     ]
    for lab, [n, y] in counts2[['No', 'Yes']].fillna(0).astype(int).iterrows():
        lab_split = check_wrapper.wrap(lab)
        append = np.hstack([f'  {y:>3d}   {n:>3d}   ' ,[''] * len(lab_split)])
        line = '\n'.join([''.join([x.ljust(spacer),y])
                          for x, y in zip(*(lab_split, append))])
        table.append(line)
    table.append(''.join(['-'] * (spacer + 15)))
    if missing_ is not None:
        table.append(missing_)

    return '\n     '.join(table)   


def _build_radio_table(count2, spacer=55):
    missing = _count_missing(count2)
    table = [
        ''.join(['     ', ''.join(['-'] * (spacer + 15))]),
        ''.join(['  Response'.ljust(spacer), '    ', 'Count']),
        ''.join(['-'] * (spacer + 15)),
    ]
    if missing is not None:
        counts2.drop(['chose not to respond'], inplace=True)
    for lab, count in count2.items():
        lab_split = check_wrapper.wrap(lab)
        append = np.hstack([f' {count:>3d} ', ['']*len(lab_split)])
        line = '\n'.join(['    '.join([x.ljust(spacer), y])
                          for x, y in zip(*(lab_split, append))])
        table.append(line)
    table.append(''.join(['-'] * (spacer + 15)))

    if missing is not None:
        table.append(missing_)

    return '\n     '.join(table)


def _build_rank_table(describe, missing, spacer=45):
    missing_ = _count_missing(num_=missing)
    table = [
        ''.join(['     ', ''.join(['-'] * (75))]),
        '   '.join(['   Answer'.ljust(spacer), 'In top 5', 'Median Rank']),
        ''.join(['-'] * (75))
     ]
    for lab, [count, med] in describe[['count', '50%']].iterrows():
        lab_split = rank_wrapper.wrap(lab)
        append = np.hstack([f'     {count:>3.0f}          {med:>3.0f}    ' ,
                            [''] * len(lab_split)])
        line = '\n'.join([''.join([x.ljust(spacer),y])
                          for x, y in zip(*(lab_split, append))])
        table.append(line)
    table.append(''.join(['-'] * (75)))
    if missing_ is not None:
        table.append(missing_)

    return '\n     '.join(table)


func_look_up = {
    'multiple choice': summarize_multiple_choice,
    'checklist': summarize_checks,
    'free text': summarize_free_text,
    'ranked': summarize_ranks,
}

def describe_column(prefix_, data_dict, data, show_notes=True, show_table=True, 
                    rename_values=True):
    """
    A tidy function to print a pretty little summary
    """
    cols = [c for c in data_dict.index if (prefix_ in c)]
    question_types = data_dict.loc[cols, 'question_type'].unique()

    if 'checkbox' in question_types:
        cols = [c for c in cols if '_[' in c]
    elif len(cols) == 0:
        raise ValueError('The item could not be found. Please check the data'
                         ' dictionary again')
    else:
        cols = [cols[0]]

    question_type = data_dict.loc[cols[0], 'question_type']
    
    # Builds the count table
    clean_func = func_look_up.get(question_type, None)
    if (clean_func is not None) & show_table:
        col_info = clean_func(prefix_, data_dict, data, rename=rename_values)
        text_table = col_info[-1]
    else:
        text_table = None
        
    # prints the summary
    summary = print_summary(cols[0], data_dict, data, counts=text_table, 
                            show_notes=show_notes)
    
    return summary