#!/usr/bin/env python3
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


def readCsv(fname):
    labels = None
    data   = None

    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        isFirstRow = True
        for row in reader:
            if isFirstRow:
                labels = row
                isFirstRow = False
                data = [[] for _ in range(len(labels))]
            else:
                for item, tgt in zip(row, data):
                    tgt.append(float(item))

    data = [np.array(x) for x in data]
    return labels, data

def main():
    parser = argparse.ArgumentParser(description='Create a plot from csv files.')

    parser.add_argument('-i', '--input', help='Csv File', required=True)
    parser.add_argument('-o', '--out', help='Output plot', required=True)
    parser.add_argument('-y', '--ylabel', help='y label')

    args = parser.parse_args()

    labels, data = readCsv(args.input)

    plt.boxplot(data, vert=True)
    plt.xticks(range(1, len(labels)+1), labels, rotation=45)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)

    if args.ylabel is not None:
        plt.ylabel(args.ylabel)

    plt.savefig(args.out)


if __name__ == '__main__':
    main()
