#!/usr/bin/env python3
import argparse
import csv
from collections import OrderedDict
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import toml


NS_TO_MS = 1000000


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


class Step:

    def __init__(self, name, queued, submitted, start, end, lane, run):

        self.name = name

        self.queued = queued
        self.submitted = submitted
        self.start = start
        self.end = end

        self.lane = lane
        self.run = run

    def normalize(self, offset):
        self.queued -= offset
        self.submitted -= offset
        self.start -= offset
        self.end   -= offset


class Measurements:

    def __init__(self, fname):
        with open(fname, 'r') as datafile:
            dct = json.load(datafile)

        self._steps = [ Step(**stepdct) for stepdct in dct["steps"] ]

        self._normalize()

    def steps(self):
        names = self._query(lambda s: s.name, unique=True)

        return names

    def runs(self):
        runIds = self._query(lambda s: s.run, unique=True)

        return runIds

    def frames(self, pred):

        startEnds = self._query(
                lambda s: (s.start, s.end), 
                pred);

        starts = [step[0] for step in startEnds]
        ends   = [step[1] for step in startEnds]

        starts = [x for x in starts if x > 0]
        ends = [x for x in ends if x > 0]

        starts = np.array(starts)
        ends = np.array(ends)

        starts = np.array(starts, dtype=np.float64)
        ends = np.array(ends, dtype=np.float64)

        starts /= NS_TO_MS
        ends /= NS_TO_MS

        return (starts, ends)

    def durations(self, pred):

        start, end = self.frames(pred)

        duration = end - start

        return list(duration)

    def _query(self, map_, pred=None, unique=False):
        
        if pred is not None:
            filtered = [ step for step in self._steps if pred(step) ]
        else:
            filtered = self._steps

        res = [map_(step) for step in filtered]

        if unique:
            return list(OrderedDict.fromkeys(res))
        else:
            return res

    def _normalize(self):

        for run in self.runs():

            print("run =", run)

            stepsByRun = self._query(
                lambda s: s, 
                lambda s: s.run == run)

            offset = np.min([s.queued for s in stepsByRun])

            print("offset =", offset)

            for step in stepsByRun:
                step.normalize(offset)


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

        durations = np.array([
                measurements.durations(lambda s: s.lane == 0 and s.name == step)
                        for step in steps])

        maxs, mins, medians = split(durations)

        maxs -= medians
        mins = medians - mins

        print(barPositions)
        print(medians)
        print(mins)
        print(maxs)

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


def timeline(cnf):
    assert len(cnf.input) == 1

    measurements = Measurements(cnf.expand(cnf.input[0].file))

    steps = measurements.steps()

    queuenum = None

    for step in steps:
        start, end = measurements.frames(
                lambda s: s.name == step and s.run == 4)

        if queuenum is None:
            queuenum = len(start)

        print(step)
        print(start)
        print(end)

        plt.barh(range(len(start)), end-start, left=start, label=step)

    assert queuenum is not None

    plt.yticks(range(queuenum))

def expandedTimeline(cnf):
    assert len(cnf.input) == 1

    measurements = Measurements(cnf.expand(cnf.input[0].file))



def finalize(cnf):
    if cnf.ylabel is not None:
        plt.ylabel(cnf.ylabel)

    if cnf.xlabel is not None:
        plt.xlabel(cnf.xlabel)

    if cnf.showLegend:
        plt.legend(loc='best')

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

    finalize(config)


if __name__ == '__main__':
    main()
