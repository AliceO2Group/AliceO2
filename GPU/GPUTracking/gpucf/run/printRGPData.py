#!/usr/bin/env python3
import argparse
import csv


COLUMNS = [
    ('Method', 40), 
    ('Time', 10), 
    ('LocalMemSize', 15), 
    ('VGPRs', 10), 
    ('SGPRs', 10),    
    # ('ScratchRegs', 15),
    # ('VALUUtilization', 25),
    ('Wavefronts', 15),
    ('VALUBusy', 10),
    ('SALUBusy', 10),
    ('L1CacheHit', 15),
    ('L2CacheHit', 15),
    ('LDSBankConflict', 20),
]


def readCsv(f):
    dreader = csv.DictReader(
            [row for row in f if not row.startswith('#')],
            skipinitialspace=True)
    data = [row for row in dreader]
    return data

def printTable(cols, data):
    colsize = [col for   _, col in cols]
    keys    = [key for key, _   in cols]
    rowfmt = ''.join(["{:>" + str(col) + "}" for col in colsize])

    print(rowfmt.format(*keys))

    for row in data:
        filteredRow = [row[key] for key in keys]
        print(rowfmt.format(*filteredRow))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print output from RGP.')
    parser.add_argument('file', metavar='CSV', type=str)

    args = parser.parse_args()

    with open(args.file) as infile:
        data = readCsv(infile)

    printTable(COLUMNS, data)
