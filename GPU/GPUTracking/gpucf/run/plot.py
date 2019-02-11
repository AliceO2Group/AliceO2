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

def split(data):
    maxs = np.array([np.max(col) for col in data])
    mins = np.array([np.min(col) for col in data])
    medians = np.array([np.median(col) for col in data])

    return maxs, mins, medians

def boxplot(args):
    print("Creating boxplot...")

    print(args.inputFiles)
    assert len(args.inputFiles) % 2 == 0

    labels, _ = readCsv(args.inputFiles[0])

    stepSize = len(args.inputFiles)
    stepNum = len(labels)

    barWidth = 0.75

    indexes = np.array(
            range(1, (stepNum * stepSize) + 1, stepSize), dtype=np.float64)

    barPositions = np.array(indexes)

    for i in range(0, len(args.inputFiles), 2):
        fname = args.inputFiles[i]
        label = args.inputFiles[i+1]
        
        print(fname, label)

        _, data = readCsv(fname)

        maxs, mins, medians = split(data)

        maxs -= medians
        mins = medians - mins

        # print(maxs)
        # print(mins)
        # print(medians)

        plt.bar(barPositions,
                medians, 
                barWidth,
                yerr=[mins,maxs],
                label=label)
        barPositions += barWidth

    plt.ylim(ymin=0)

    plt.xticks(indexes + barWidth / 2, labels, rotation=20)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)

    if args.ylabel is not None:
        plt.ylabel(args.ylabel)

    plt.legend()

    plt.savefig(args.out)

def main():
    parser = argparse.ArgumentParser(description='Create a plot from csv files.')

    subparsers = parser.add_subparsers()

    boxplotParser = subparsers.add_parser('boxplot', help="Create boxplots")
    boxplotParser.set_defaults(func=boxplot)
    boxplotParser.add_argument(
            'inputFiles', 
            metavar='CSV LABEL', 
            nargs='+', 
            help='Input files')
    boxplotParser.add_argument('-o', '--out', help='Output plot', required=True)
    boxplotParser.add_argument('-y', '--ylabel', help='y label')

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
