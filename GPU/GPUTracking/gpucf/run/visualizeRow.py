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
MAX_TIME = 1000
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



def readDigits(fName):
    digits = []
    with open(fName, 'r') as digitFile:
        for line in digitFile:
            match = re.match(DIGIT_REGEX, line)

            if match is None:
                continue

            charge = float(match.group(1))
            cru = int(match.group(2))
            row = int(match.group(3))
            row = GLOBAL_TO_LOCAL_ROW[row]
            pad = int(match.group(4))
            time = int(match.group(5))
            digits.append(Digit(charge, cru, row, pad, time))
    return digits


def readClusters(fName):
    clusters = []
    with open(fName, 'r') as clusterFile:
        for line in clusterFile:
            match = re.match(CLUSTER_REGEX, line)

            if match is None:
                continue

            cru = int(match.group(1))
            row = int(match.group(2))
            Q = float(match.group(3))
            QMax = float(match.group(4))
            padMean = float(match.group(5))
            timeMean = float(match.group(6))
            padSigma = float(match.group(7))
            timeSigma = float(match.group(8))

            clusters.append(Cluster(cru, row, Q, QMax, padMean, 
                timeMean, padSigma, timeSigma))

    return clusters


def filterByRow(clusters, cru, localRow):
    return list(filter(lambda c: c.cru == cru and c.row == localRow, clusters))


class GridCanvas:

    def __init__(self, widthOfCell, heightOfCell, cellsPerRow, cellsPerColumn):
        assert heightOfCell > 0
        assert widthOfCell > 0
        assert cellsPerRow > 0
        assert cellsPerColumn > 0

        self.heightOfCell = heightOfCell
        self.widthOfCell = widthOfCell
        self.cellsPerRow = cellsPerRow
        self.cellsPerColumn = cellsPerColumn
        self.gridLineDiameter = 1

        assert self.heightOfCell > self.gridLineDiameter
        assert self.widthOfCell > self.gridLineDiameter

        self.image = QImage(self.widthPx(), 
                            self.heightPx(), 
                            QImage.Format_RGB32)
        self.painter = QPainter(self.image)

    def widthPx(self):
        return self.widthOfCell * self.cellsPerRow

    def heightPx(self):
        return self.heightOfCell * self.cellsPerColumn

    def fill(self, col: QColor):
        self.image.fill(col)
        self._drawGrid()

    def _drawGrid(self):
        self.painter.setPen(QColor('black'))
        for x in range(0, self.widthPx(), self.widthOfCell):
            self.painter.drawLine(x, 0, x, self.heightPx())
        for y in range(0, self.heightPx(), self.heightOfCell):
            self.painter.drawLine(0, y, self.widthPx(), y)

    def fillCell(self, x, y, col: QColor):
        assert x >= 0 and x < self.cellsPerRow
        assert y >= 0 and y < self.cellsPerColumn
        
        self.painter.fillRect(x*self.widthOfCell+1, 
                              y*self.heightOfCell+1,
                              self.widthOfCell - 1,
                              self.heightOfCell - 1,
                              col)

    def drawEllipse(self, x, y, width, height, col: QColor):
        x *= self.widthOfCell
        y *= self.heightOfCell
        width *= self.widthOfCell
        height *= self.heightOfCell

        self.painter.setPen(col)
        self.painter.drawEllipse(QPointF(x, y), width, height)

    def drawDigits(self, digits, col: QColor):
        for digit in digits:
            self.fillCell(digit.time, digit.pad, col)

    def drawClusters(self, clusters, col: QColor):
        for cluster in clusters:
            self.drawEllipse(cluster.timeMean, 
                             cluster.padMean, 
                             cluster.timeSigma,
                             cluster.padSigma,
                             col)

    def save(self, fname):
        self.painter.end()
        self.image.save(fname)


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
    parser.add_argument('-p', '--padPx', type=int, help='Pixel per pad', default=16)
    parser.add_argument('-t', '--timePx', type=int, help='Pixel per time slice', default=16)

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

    grid = GridCanvas(args.timePx, args.padPx, MAX_TIME, PADS_PER_ROW) 
    grid.fill(QColor('white'))

    print('Drawing {} digits...'.format(len(digits)))
    grid.drawDigits(digits, DIGIT_COLOR)

    if args.peaks is not None:
        print('Drawing peaks...')
        peaks = readDigits(args.peaks)
        peaks = filterByRow(peaks, args.cru, args.row)
        grid.drawDigits(peaks, PEAK_COLOR)

    grid.drawClusters(calcClusters, GPU_CLUSTER_COLOR)
    grid.drawClusters(truthClusters, TRUTH_CLUSTER_COLOR)

    grid.save(args.out)


if __name__ == '__main__':
    main()
