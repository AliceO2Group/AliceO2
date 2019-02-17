#!/usr/bin/env python3
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import toml


def requireKey(dict_, key):
    if key not in dict_:
        raise Exception("Required key {} missing.".format(key))


class Input:

    requiredKeys = [
        "file",
        "label"
    ]

    def __init__(self, dict_):
        for req in Input.requiredKeys:
            requireKey(dict_, req)
            self.__dict__[req] = dict_[req]


class Config:

    requiredKeys = [
        "type",
        "out",
        "input"
    ]

    optionalKeys = [
        "ylabel",
        "showLegend"
    ]

    def __init__(self, confDict, baseDir):

        self.baseDir = baseDir

        print(confDict)
        for req in Config.requiredKeys:
            requireKey(confDict, req)

        self.type = confDict["type"]
        self.out  = confDict["out"]

        self.input = [Input(dict_) for dict_ in confDict["input"]]
        assert self.input

        assert all([(req in self.__dict__) for opt in Config.requiredKeys])

        for opt in Config.optionalKeys:
            self.__dict__[opt] = confDict.get(opt)

        assert all([(opt in self.__dict__) for opt in Config.optionalKeys])

    def expand(self, fname):
        return os.path.join(self.baseDir, fname)


class Measurements:

    def __init__(self, fname):
        with open(fname, 'r') as datafile:
            dct = json.load(datafile)
        # print(dct)
        self.data = dct["runs"]

    def labels(self):
        return [step["name"] for step in self.data[0]["lanes"][0]]

    def lanes(self):
        return len(self.data[0]["lanes"])

    def runs(self):
        return len(self.data)

    def durations(self, lane=0):

        # print("runs =", self.runs())
        # print("lanes =", self.lanes())

        start = np.array([
            [ self.data[run]["lanes"][lane][i]["start"] 
                for run in range(self.runs()) ]
            for i in range(len(self.labels()))
        ])

        # print("start =", start)

        end = np.array([
            [ self.data[run]["lanes"][lane][i]["end"] 
                for run in range(self.runs()) ]
            for i in range(len(self.labels()))
        ])

        # print("end = ", end)

        duration = end - start

        duration = np.array(duration, dtype=np.float64)
        duration /=  1000000

        # print("duration =", duration)

        return list(duration)


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

    print(maxs, mins, medians)

    return maxs, mins, medians

def bar(cnf):
    print("Creating barplot...")

    measurements = Measurements(cnf.expand(cnf.input[0].file))

    labels = measurements.labels()

    stepSize = len(cnf.input)
    stepNum = len(labels)

    barWidth = 0.75

    indexes = np.array(
            range(1, (stepNum * stepSize) + 1, stepSize), dtype=np.float64)

    barPositions = np.array(indexes)

    for i in cnf.input:
        fname = cnf.expand(i.file)
        label = i.label
        
        print(fname, label)

        measurements = Measurements(fname)

        maxs, mins, medians = split(measurements.durations())

        maxs -= medians
        mins = medians - mins

        plt.bar(barPositions,
                medians, 
                barWidth,
                yerr=[mins,maxs],
                label=label)
        barPositions += barWidth

    plt.ylim(ymin=0)

    plt.xticks(indexes + (stepSize - 1) * barWidth / 2, labels, rotation=20)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)

    if cnf.ylabel is not None:
        plt.ylabel(cnf.ylabel)

    if cnf.showLegend:
        plt.legend()

    plt.savefig(cnf.expand(cnf.out))


PLOTS = {
        "bar" : bar
}


def main():
    parser = argparse.ArgumentParser(description='Create a plot from csv files.')

    parser.add_argument(
            'config', 
            metavar='CNF', 
            nargs=1,
            help='Config file describing the plot')

    args = parser.parse_args()

    config = Config(toml.load(args.config), os.path.dirname(args.config[0]))

    PLOTS[config.type](config)


if __name__ == '__main__':
    main()
