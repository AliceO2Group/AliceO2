#!/usr/bin/env python3
import argparse
import collections
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtCore    import *
from PyQt5.QtGui     import *
from PyQt5.QtWidgets import *


PADDING = 2
NUM_OF_ROWS = 152
PADS_PER_ROW = 138
PADS_PER_ROW_PADDED = PADS_PER_ROW + 2*PADDING
MAX_TIME = 1400
MAX_TIME_PADDED = MAX_TIME + 2*PADDING

DIGIT_COLOR = QColor('cornflowerblue')
PEAK_COLOR  = QColor('darkblue')
TRUTH_CLUSTER_COLOR = QColor('coral')
GPU_CLUSTER_COLOR = QColor('crimson')


def matchField(name):
    return r'{}\s*=\s*([^,\s]*)'.format(name)

def matchStruct(name, fields):
    prefix = r'{}:\s*'.format(name)
    fields = r'\s*,\s*'.join(map(matchField, fields))
    return prefix + fields


DIGIT_REGEX = matchStruct('Digit', ['charge', 'cru', 'row', 'pad', 'time'])
CLUSTER_REGEX = matchStruct('Cluster', ['cru', 'row', 'Q', 'Qmax', 
                                        'padMean', 'timeMean', 
                                        'padSigma', 'timeSigma'])

Cluster = collections.namedtuple('Cluster', 
        ['cru', 'row', 'Q', 'QMax', 'padMean', 'timeMean', 'padSigma', 'timeSigma'])

Digit = collections.namedtuple('Digit', 
        ['charge', 'cru', 'row', 'pad', 'time'])

def makeGlobalToLocalRow(rowsPerRegion):
    globalToLocalRow = []
    for rows in rowsPerRegion:
        for local in range(rows):
            globalToLocalRow.append(local)
    return globalToLocalRow

GLOBAL_TO_LOCAL_ROW = makeGlobalToLocalRow([17, 15, 16, 15, 18, 16, 16, 14, 13, 12])


def readAsDict(fName):
    dicts = []
    with open(fName, 'r') as infile:
        for line in infile:
            line = ''.join(line.split())
            typeAndFields = line.split(':')
            if len(typeAndFields) != 2:
                continue
            res = {}
            type_ = typeAndFields[0]
            res['__type__'] = type_
            fields = typeAndFields[1]
            ok = True
            for keyValPair in fields.split(','):
                keyAndVal = keyValPair.split('=')
                if len(keyAndVal) != 2:
                    ok = False
                    break
                key = keyAndVal[0]
                val = keyAndVal[1]
                res[key] = val
            if ok:
                dicts.append(res)
    return dicts

def readDigits(fName):
    digits = []
    for dict_ in readAsDict(fName):
        charge = float(dict_['charge'])
        cru = int(dict_['cru'])
        row = int(dict_['row'])
        row = GLOBAL_TO_LOCAL_ROW[row]
        pad = int(dict_['pad'])
        time = int(dict_['time'])
        digits.append(Digit(charge, cru, row, pad, time))
    return digits


def readClusters(fName):
    clusters = []
    for dict_ in readAsDict(fName):
        cru = int(dict_['cru'])
        row = int(dict_['row'])
        Q = float(dict_['Q'])
        QMax = float(dict_['QMax'])
        padMean = float(dict_['padMean'])
        timeMean = float(dict_['timeMean'])
        padSigma = float(dict_['padSigma'])
        timeSigma = float(dict_['timeSigma'])

        clusters.append(Cluster(cru, row, Q, QMax, padMean, 
            timeMean, padSigma, timeSigma))

    return clusters


def filterByRow(clusters, cru, localRow):
    return list(filter(lambda c: c.cru == cru and c.row == localRow, clusters))


class GridCanvas:

    def __init__(self, startPad, startTime, padWidth, timeWidth, 
            cellPadWidth, cellTimeWidth):

        self.startPad = startPad
        self.startTime = startTime
        self.padWidth = padWidth
        self.timeWidth = timeWidth
        self.cellPadWidth = cellPadWidth
        self.cellTimeWidth = cellTimeWidth
        self.gridLineDiameter = 1

        self.image = QImage(self.widthPx(), 
                            self.heightPx(), 
                            QImage.Format_RGB32)
        self.painter = QPainter(self.image)

    def widthPx(self):
        return self.cellTimeWidth * self.timeWidth

    def heightPx(self):
        return self.cellPadWidth * self.padWidth

    def fill(self, col: QColor):
        self.image.fill(col)
        self._drawGrid()

    def _drawGrid(self):
        self.painter.setPen(QColor('black'))
        for x in range(0, self.widthPx(), self.cellTimeWidth):
            x -= self.cellPadWidth/2
            self.painter.drawLine(x, 0, x, self.heightPx())
        for y in range(0, self.heightPx(), self.cellPadWidth):
            y -= self.cellTimeWidth/2
            self.painter.drawLine(0, y, self.widthPx(), y)

    def fillCell(self, pad, time, col: QColor):
        pad -= self.startPad
        time -= self.startTime

        self.painter.fillRect(pad*self.cellPadWidth+1-self.cellPadWidth/2,
                              time*self.cellTimeWidth+1-self.cellTimeWidth/2, 
                              self.cellPadWidth - 1,
                              self.cellTimeWidth - 1,
                              col)

    def drawEllipse(self, padCenter, timeCenter, padWidth, timeWidth, 
            col: QColor):
        padCenter -= self.startPad
        timeCenter -= self.startTime
        padCenter *= self.cellPadWidth
        timeCenter *= self.cellTimeWidth
        padWidth *= self.cellPadWidth
        timeWidth *= self.cellTimeWidth

        self.painter.setPen(col)
        self.painter.drawEllipse(
                QPointF(padCenter, timeCenter), 
                padWidth, timeWidth)

    def drawDigits(self, digits, col: QColor):
        for digit in digits:
            if self.pointInGrid(digit.pad, digit.time):
                print(digit)
                self.fillCell(digit.pad, digit.time, col)

    def drawClusters(self, clusters, col: QColor):
        for cluster in clusters:
            if self.pointInGrid(cluster.padMean, cluster.timeMean):
                print(cluster)
                self.drawEllipse(cluster.padMean, 
                                 cluster.timeMean, 
                                 cluster.padSigma,
                                 cluster.timeSigma,
                                 col)

    def save(self, fname):
        self.painter.end()
        self.image.save(fname)

    def pointInGrid(self, pad, time):
        return pad >= self.startPad \
           and time >= self.startTime \
           and pad < self.startPad + self.padWidth \
           and time < self.startTime + self.timeWidth


def main():
    parser = argparse.ArgumentParser(description="Draw a row of digits.")
    parser.add_argument('-df', '--digits', help='File of digits',
            required=True)
    parser.add_argument('-pf', '--peaks', help='File of peaks')
    parser.add_argument('-tf', '--truth', help='File of ground truth clusters',
            required=True)
    parser.add_argument('-cf', '--clusters', help='File of calculated clusters', 
            required=True)
    parser.add_argument('-r', '--row', type=int, help='local row to paint', required=True)
    parser.add_argument('-c', '--cru', type=int, help='cru of local row', required=True)
    parser.add_argument('-o', '--out', help='outfile', required=True)
    parser.add_argument('-p', '--pad', type=int, help='Start pad', required=True)
    parser.add_argument('-t', '--time', type=int, help='Start time', required=True)
    parser.add_argument('-pw', '--padWidth', type=int, help='pad width', required=True)
    parser.add_argument('-tw', '--timeWidth', type=int, help='time width', required=True)
    parser.add_argument('-px', '--padPx', type=int, help='Pixel per pad', default=16)
    parser.add_argument('-tx', '--timePx', type=int, help='Pixel per time slice', default=16)

    app = QApplication(sys.argv)

    args = parser.parse_args()

    print("Reading cluster file '{}' and truth file '{}'.".format(
        args.clusters, args.truth))

    calcClusters = readClusters(args.clusters)
    truthClusters = readClusters(args.truth)
    digits = readDigits(args.digits)

    print("Found {} clusters and {} ground truth clusters.".format(
        len(calcClusters), len(truthClusters)))

    print("Painting cru {}, row {}".format(args.cru, args.row))
    calcClusters = filterByRow(calcClusters, args.cru, args.row)
    truthClusters = filterByRow(truthClusters, args.cru, args.row)
    digits = filterByRow(digits, args.cru, args.row)

    print("Found {} clusters and {} ground truth clusters in this row.".format(
        len(calcClusters), len(truthClusters)))

    grid = GridCanvas(args.pad, args.time, 
                      args.padWidth, args.timeWidth, 
                      args.timePx, args.padPx)
    grid.fill(QColor('white'))

    print('Drawing {} digits...'.format(len(digits)))
    grid.drawDigits(digits, DIGIT_COLOR)

    if args.peaks is not None:
        print('Drawing peaks...')
        peaks = readDigits(args.peaks)
        peaks = filterByRow(peaks, args.cru, args.row)
        grid.drawDigits(peaks, PEAK_COLOR)

    print('Drawing cluster...')
    grid.drawClusters(calcClusters, GPU_CLUSTER_COLOR)
    print('Drawing ground truth...')
    grid.drawClusters(truthClusters, TRUTH_CLUSTER_COLOR)

    grid.save(args.out)


if __name__ == '__main__':
    main()
