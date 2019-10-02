from itertools import islice
import os
import csv
import pandas as pd
import io


def reversed_lines(file):
    "Generate the lines of file in reverse order."
    part = ''
    for block in reversed_blocks(file):
        for c in reversed(block):
            if c == '\n' and part:
                yield part[::-1]
                part = ''
            part += c
    if part: yield part[::-1]


def reversed_blocks(file, blocksize=4096):
    "Generate blocks of file's contents in reverse order."
    file.seek(0, os.SEEK_END)
    here = file.tell()
    while 0 < here:
        delta = min(blocksize, here)
        here -= delta
        file.seek(here, os.SEEK_SET)
        yield file.read(delta)

def CsvEndReader(path, lines, Processed = False):
    out = []
    with open(path) as file:
        for line in islice(reversed_lines(file), lines):
            reader = csv.reader(io.StringIO(line), delimiter=',')
            out.append(next(reader))
    if Processed:
        out = pd.DataFrame(out).set_index(0)
    else:
        out = pd.DataFrame(out, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp')

    return out[::-1]

