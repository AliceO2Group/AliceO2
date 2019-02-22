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
        "xlabel",
        "showLegend"
    ]

    def __init__(self, confDict, baseDir):

        self.baseDir = baseDir

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


class Lane:

    def __init__(self, data):
        self.orderedNames = []

        self.data = {}

        for step in data:
            self.orderedNames.append(step["name"])
            self.data[step["name"]] = (step["start"], step["end"])

    def starts(self):
        return [step[0] for step in self.flatten()]

    def ends(self):
        return [step[1] for step in self.flatten()]

    def names(self):
        return self.orderedNames

    def flatten(self):
        return [self.data[name] for name in self.orderedNames]

    def step(self, name):
        return self.data[name]

    def begin(self):
        return self.data[self.orderedNames[0]][0]

class Run:

    def __init__(self, data):
        self.start = data["start"]
        self.end   = data["end"]

        self.lanes = [Lane(dct) for dct in data["lanes"]]

    def begin(self):
        begins = [lane.begin() for lane in self.lanes]

        return min(begins)


class Measurements:

    def __init__(self, fname):
        with open(fname, 'r') as datafile:
            dct = json.load(datafile)

        self.data = dct["runs"]

        self.runs = []

        for rundct in dct["runs"]:
            self.runs.append(Run(rundct))


    def steps(self):
        return [step["name"] for step in self.data[0]["lanes"][0]]

    def lanes(self):
        return len(self.data[0]["lanes"])

    def frames(self, step):

        run = 0

        lanes = self.runs[run].lanes

        frames = []

        for lane in lanes:
            frames.append(lane.step(step))

        starts = [step[0] for step in frames]
        ends   = [step[1]  for step in frames]


        starts = [x for x in starts if x > 0]
        ends = [x for x in ends if x > 0]

        starts   = np.array(starts)
        ends   = np.array(ends)

        offset = self.runs[run].begin()
        starts -= offset
        ends   -= offset

        starts = np.array(starts, dtype=np.float64)
        ends = np.array(ends, dtype=np.float64)

        starts /= 1000000.0
        ends /= 1000000.0

        return (starts, ends)

    def durations(self, lane=0):

        steps = self.steps()

        start = np.array([
            [ self.data[run]["lanes"][lane][i]["start"] 
                for run in range(len(self.runs)) ]
            for i in range(len(steps))
        ])

        end = np.array([
            [ self.data[run]["lanes"][lane][i]["end"] 
                for run in range(len(self.runs)) ]
            for i in range(len(steps))
        ])

        duration = end - start

        duration = np.array(duration, dtype=np.float64)
        duration /=  1000000

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

    return maxs, mins, medians

def bar(cnf):
    print("Creating barplot...")

    measurements = Measurements(cnf.expand(cnf.input[0].file))

    steps = measurements.steps()

    stepSize = len(cnf.input)
    stepNum = len(steps)

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

    plt.xticks(indexes + (stepSize - 1) * barWidth / 2, steps, rotation=20)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)

    if cnf.ylabel is not None:
        plt.ylabel(cnf.ylabel)

    if cnf.xlabel is not None:
        plt.label(cnf.xlabel)

    if cnf.showLegend:
        plt.legend()

    plt.savefig(cnf.expand(cnf.out))


def timeline(cnf):
    assert len(cnf.input) == 1

    measurements = Measurements(cnf.expand(cnf.input[0].file))

    steps = measurements.steps()

    for step in steps:
        start, end = measurements.frames(step)

        plt.barh(range(len(start)), end-start, left=start, label=step)


    if cnf.ylabel is not None:
        plt.ylabel(cnf.ylabel)

    if cnf.xlabel is not None:
        plt.xlabel(cnf.xlabel)

    if cnf.showLegend:
        plt.legend()

    plt.savefig(cnf.expand(cnf.out))



PLOTS = {
        "bar"      : bar,
        "timeline" : timeline
}


def main():
    parser = argparse.ArgumentParser(description='Create a plot from csv files.')

    parser.add_argument(
            'config', 
            metavar='CNF', 
            nargs=1,
            help='Config file describing the plot')

    args = parser.parse_args()

    print("Opening config file ", args.config)

    config = Config(toml.load(args.config), os.path.dirname(args.config[0]))

    PLOTS[config.type](config)


if __name__ == '__main__':
    main()
