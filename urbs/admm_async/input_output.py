import io
from os.path import join

import pandas as pd


def words_in_line(line):
    if line.endswith('\n'):
        line = line[:-1]
    return line.split(' ')


def read_results(f: io.TextIOWrapper) -> pd.DataFrame:
    lines = f.readlines()
    n_headers = len(words_in_line(lines[0]))
    skiprows = []
    for i, line in enumerate(lines):
        if line.endswith('\n'):
            line = line[:-1]
        words = line.split(' ')
        if len(words) != n_headers:
            skiprows.append(i)
            raise RuntimeWarning(f'Results file contains malformed line: line {i}')
    f.seek(0)
    df = pd.read_csv(
        f,
        sep=' ',
        header=0,
        skiprows=skiprows
    )
    return df
