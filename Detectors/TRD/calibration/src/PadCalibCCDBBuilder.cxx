// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PadCalibCCDBBuilder.cxx
/// \brief Krypton calibration - class to smoothen and inter/extrapolate gain in chambers and calculate normalized ADC gain per chamber
/// \author Jana Crkovska

#include "TRDCalibration/PadCalibCCDBBuilder.h"
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include "TFile.h"
#include "TH2.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"

using namespace o2::trd::constants;

namespace o2::trd
{

void PadCalibCCDBBuilder::checkIfIsolatedHotPadCandidate(TH2F* hDet, std::vector<int> coordinates, float upperLimit, int areaContainedWithin)
{
  auto numberOfCoordinates = coordinates.size();
  if (numberOfCoordinates != 4) {
    std::cerr << "Invalid size of the coordinates vector!" << std::endl;
    return;
  }
  float averageGain = computeDetectorAverage(hDet);

  if (hDet->GetBinContent(coordinates[0], coordinates[1]) > upperLimit * averageGain) {
    int sizeOfHotArea = isolatedHotPadsContainmentSize(hDet, coordinates[0], coordinates[1]);
    if (sizeOfHotArea > 0 && sizeOfHotArea < areaContainedWithin) {
      replaceIsolatedHotPads(hDet, coordinates[0], coordinates[1], sizeOfHotArea);
    }
  } else if (hDet->GetBinContent(coordinates[2], coordinates[3]) > upperLimit * averageGain) {
    int sizeOfHotArea = isolatedHotPadsContainmentSize(hDet, coordinates[2], coordinates[3]);
    if (sizeOfHotArea > 0 && sizeOfHotArea < areaContainedWithin) {
      replaceIsolatedHotPads(hDet, coordinates[2], coordinates[3], sizeOfHotArea);
    }
  } else if (hDet->GetBinContent(coordinates[0], coordinates[1]) == 0 && hDet->GetBinContent(coordinates[2], coordinates[3]) != 0) {
    int sizeOfHotArea = isolatedHotPadsContainmentSize(hDet, coordinates[2], coordinates[3]);
    if (sizeOfHotArea > 0 && sizeOfHotArea < areaContainedWithin) {
      replaceIsolatedHotPads(hDet, coordinates[2], coordinates[3], sizeOfHotArea);
    }
  }
}

void PadCalibCCDBBuilder::checkIfSmallerCloserToCenter(TH2F* hDet, std::vector<int> coordinates, float allowedDifference)
{
  if (hDet->GetBinContent(coordinates[0], coordinates[1]) == 0 || hDet->GetBinContent(coordinates[2], coordinates[3]) == 0) {
    return;
  }

  float xCenter = hDet->GetNbinsX() / 2.;
  float yCenter = hDet->GetNbinsY() / 2.;

  std::vector<float> vCenter, vPad1, vPad2;
  vCenter.push_back(hDet->GetNbinsX() / 2.);
  vCenter.push_back(hDet->GetNbinsY() / 2.);

  vPad1.push_back(hDet->GetXaxis()->GetBinCenter(coordinates[0]));
  vPad1.push_back(hDet->GetYaxis()->GetBinCenter(coordinates[1]));

  vPad2.push_back(hDet->GetXaxis()->GetBinCenter(coordinates[2]));
  vPad2.push_back(hDet->GetYaxis()->GetBinCenter(coordinates[3]));

  float dist1 = computeDistance(vPad1, vCenter);
  float dist2 = computeDistance(vPad2, vCenter);

  if ((dist1 < dist2) && ((hDet->GetBinContent(coordinates[2], coordinates[3]) - hDet->GetBinContent(coordinates[0], coordinates[1])) > allowedDifference)) {
    replacePadCloserToCenter(hDet, coordinates[0], coordinates[1]);
  }

  if ((dist1 > dist2) && ((hDet->GetBinContent(coordinates[0], coordinates[1]) - hDet->GetBinContent(coordinates[2], coordinates[3])) > allowedDifference)) {
    replacePadCloserToCenter(hDet, coordinates[2], coordinates[3]);
  }
}

std::vector<int> PadCalibCCDBBuilder::compareGain(TH2F* hDet, int column, int row, int shiftcolumn, int shiftrow, float allowedDifference)
{
  std::vector<int> coordinates = {-1, -1, -1, -1};

  int colMax = hDet->GetNbinsX();
  int rowMax = hDet->GetNbinsY();

  // only allow shift along one axis and only by one pad
  if (!(shiftcolumn == 1 && shiftrow == 0) && !(shiftcolumn == 0 && shiftrow == 1)) {
    return coordinates;
  }

  // cheks that the pad is valid
  if ((column >= colMax) || (row >= rowMax)) {
    return coordinates;
  }

  if ((column == colMax && shiftcolumn == 1) || (row == rowMax && shiftrow == 1)) {
    return coordinates; // do not compare with overflow
  }

  float gain1 = hDet->GetBinContent(column, row);
  float gain2 = hDet->GetBinContent(column + shiftcolumn, row + shiftrow);

  if ((gain1 == 0 && gain2 > 0) || (gain1 > 0 && gain2 == 0) || (abs(gain1 - gain2) > allowedDifference)) {
    coordinates[0] = column;
    coordinates[1] = row;
    coordinates[2] = column + shiftcolumn;
    coordinates[3] = row + shiftrow;
  }

  return coordinates;
}

float PadCalibCCDBBuilder::computeDetectorAverage(TH2F* hDet)
{ // computes an average over filled cells
  // cells are accessed through their values, not bin coordinates
  // the average os computed over absolute values as inter/extrapolated cells
  // have negatiove values

  float average = 0.;
  int nBinsUsed = 0;
  for (int irow = 0; irow < hDet->GetNbinsY(); irow++) {
    for (int icol = 0; icol < hDet->GetNbinsX(); icol++) {
      float currentGain = TMath::Abs(hDet->GetBinContent(icol + 1, irow + 1));
      if (currentGain == 0) {
        continue;
      }
      average += currentGain;
      nBinsUsed++;
    }
  }
  average /= nBinsUsed;

  return average;
}

float PadCalibCCDBBuilder::computeDistance(std::vector<float> pad1, std::vector<float> pad2)
{
  float distance = -1.;

  auto numberOfCoordinates = pad1.size();
  if (numberOfCoordinates != pad2.size()) {
    std::cerr << "Something fishy with the pad coordinates!" << std::endl;
    return distance;
  }

  for (int i = 0; i < numberOfCoordinates; i++) {
    distance += TMath::Power(pad1[i] - pad2[i], 2);
  }
  distance = TMath::Sqrt(distance);

  return distance;
}

TH2F* PadCalibCCDBBuilder::createNormalizedMap(TH2F* hDet, TString sNewName)
{                       // clones the filled 2d map
                        // computes an average
                        // replaces each bin conetnt with ratio of gain over the average
  if (sNewName == "") { // create a default name
    sNewName = hDet->GetName();
    sNewName += "_normalized";
  }
  TH2F* hDetNormalized = (TH2F*)hDet->Clone(sNewName.Data());

  float average = computeDetectorAverage(hDet);

  for (int irow = 0; irow < hDet->GetNbinsY(); irow++) {
    for (int icol = 0; icol < hDet->GetNbinsX(); icol++) {
      float currentGain = hDet->GetBinContent(hDet->GetXaxis()->FindBin(icol), hDet->GetYaxis()->FindBin(irow));
      float newGain = currentGain / average;
      hDetNormalized->SetBinContent(hDetNormalized->GetXaxis()->FindBin(icol), hDetNormalized->GetYaxis()->FindBin(irow), newGain);
    }
  }

  return hDetNormalized;
}

void PadCalibCCDBBuilder::fillInTheGap(TH2F* hDet, int column, int row, float newGain)
{ // takes hDet and replaces content of bin (column,row) with gain
  // CHANGES THE TH2 GIVEN IN ARGUMENT!
  float currentGain = hDet->GetBinContent(column + 1, row + 1);
  // float currentGain = hDet->GetBinContent(hDet->GetXaxis()->FindBin(column), hDet->GetYaxis()->FindBin(row));
  if (currentGain != 0) {
    return; // make sure we don't replace gain, just fill in gaps
  }
  float factor = 1;
  hDet->SetBinContent(column + 1, row + 1, factor * newGain);
  // hDet->SetBinContent(hDet->GetXaxis()->FindBin(column), hDet->GetYaxis()->FindBin(row), factor*newGain);
}

TH2F* PadCalibCCDBBuilder::fillTheMap(TH2F* hDet, TString sNewName, int nbuffer)
{                       // clones the map and fills the clone
                        // will fill any map with at least 1 hit!
                        // the map is cloned to limit propagation of neighboring bin content
  if (sNewName == "") { // create default name
    sNewName = hDet->GetName();
    sNewName += "_filled";
  }
  TH2F* hDetFilled = (TH2F*)hDet->Clone(sNewName);

  TH2F* hDetTemp = (TH2F*)hDet->Clone("hDetTemp"); // use as intermediate det in the filling process
  // find empty bins
  std::vector<std::vector<int>> emptyBinsColRow = findEmpty(hDetTemp);
  // loop over empty bins of th clone and fill them with the average gain
  // calculated from the gain in neighboring bins in the "Temp" map
  auto nEmptyBins = emptyBinsColRow.size();

  int firstFilledX = hDetFilled->FindFirstBinAbove();
  int lastFilledX = hDetFilled->FindLastBinAbove();
  int nBinsX = hDetFilled->GetNbinsX();

  int firstFilledY = hDetFilled->FindFirstBinAbove(0, 2);
  int lastFilledY = hDetFilled->FindLastBinAbove(0, 2);
  int nBinsY = hDetFilled->GetNbinsY();

  while (nEmptyBins != 0) {

    for (int ibin = 0; ibin < nEmptyBins; ibin++) {
      // printf("filling bin (%i:%i) \n", emptyBinsColRow[ibin][0], emptyBinsColRow[ibin][1]);
      int flippedCoordinateX = nBinsX - hDetFilled->GetXaxis()->FindBin(emptyBinsColRow[ibin][0]) + 1;
      int flippedCoordinateY = nBinsY - hDetFilled->GetYaxis()->FindBin(emptyBinsColRow[ibin][1]) + 1;
      float mirroredGain = hDetFilled->GetBinContent(flippedCoordinateX, hDetFilled->GetYaxis()->FindBin(emptyBinsColRow[ibin][1]));
      //
      if (mirroredGain == 0) {
        mirroredGain = hDetFilled->GetBinContent(hDetFilled->GetXaxis()->FindBin(emptyBinsColRow[ibin][0]), flippedCoordinateY);
      }
      //
      if (mirroredGain == 0) {
        mirroredGain = getAverageFromNeighbors(hDetTemp, emptyBinsColRow[ibin][0], emptyBinsColRow[ibin][1], nbuffer);
      }
      //
      float factor = -1;
      if (mirroredGain < 0) {
        factor = 1;
      }
      fillInTheGap(hDetFilled, emptyBinsColRow[ibin][0], emptyBinsColRow[ibin][1], factor * mirroredGain);
    }
    //
    auto nEmptyPrevious = emptyBinsColRow.size();
    emptyBinsColRow.clear();
    hDetTemp = (TH2F*)hDetFilled->Clone("hDetTemp");
    emptyBinsColRow = findEmpty(hDetTemp);
    nEmptyBins = emptyBinsColRow.size();
    if (nEmptyPrevious == nEmptyBins) {
      break; // will break out of the loop if no more empty pads can be filled
    }
  } // will continue the loop till all bins are filled

  delete hDetTemp;

  return hDetFilled;
}

std::vector<std::vector<int>> PadCalibCCDBBuilder::findEmpty(TH2F* hDetectorMap)
{ // finds the coordinates (col,row) of all empty bins
  std::vector<std::vector<int>> emptyBins;

  for (int irow = 0; irow < hDetectorMap->GetNbinsY(); irow++) {
    for (int icolumn = 0; icolumn < hDetectorMap->GetNbinsX(); icolumn++) {
      float gain = hDetectorMap->GetBinContent(icolumn + 1, irow + 1);
      if (gain == 0) {
        std::vector<int> coordinates; // todo fixed length and then set col/row as std::vector[0/1] = xx?
        coordinates.push_back(icolumn);
        coordinates.push_back(irow);
        emptyBins.push_back(coordinates);
      }
    } // loop over columns
  }   // loop over rows
  return emptyBins;
}

std::vector<std::vector<int>> PadCalibCCDBBuilder::findInhomogeneities(TH2F* hDet, float allowedDifference)
{ // finds bins that have 1+ neighbour with significantly different content
  // gives the coordonates of the pair
  std::vector<std::vector<int>> suspiciousBinPairs;

  // check for edges - this operation goes before filling
  // edges along X:
  int xFirst = hDet->FindFirstBinAbove();
  int xLast = hDet->FindLastBinAbove(); // stop the process at penultimate vs ultimate row/column
  // // edges along Y:
  int yFirst = hDet->FindFirstBinAbove(0, 2);
  int yLast = hDet->FindLastBinAbove(0, 2); // stop the process at penultimate vs ultimate row/column

  // int nbins_x = hDet->GetNbinsX();
  // int nbins_y = hDet->GetNbinsY();
  for (int irow = 0; irow < hDet->GetNbinsY(); irow++) {

    int thisRow = hDet->GetXaxis()->FindBin(irow);
    // if( thisRow < yFirst || thisRow > yLast ) continue;

    for (int icolumn = 0; icolumn < hDet->GetNbinsX(); icolumn++) {

      int thisColumn = hDet->GetXaxis()->FindBin(icolumn);
      // if( thisColumn < xFirst || thisColumn > xLast ) continue;

      std::vector<int> pair = compareGain(hDet, thisColumn, thisRow, 1, 0, allowedDifference);
      if (!(std::any_of(pair.begin(), pair.end(), [](int i) { return i == -1; }))) {
        suspiciousBinPairs.push_back(pair);
      }
      pair.clear();

      pair = compareGain(hDet, thisColumn, thisRow, 0, 1, allowedDifference);
      if (!(std::any_of(pair.begin(), pair.end(), [](int i) { return i == -1; }))) {
        suspiciousBinPairs.push_back(pair);
      }

    } // loop over columns

  } // loop over rows

  return suspiciousBinPairs;
}

float PadCalibCCDBBuilder::getAverageFromNeighbors(TH2F* hDet, int column, int row, int nbuffer)
{ // takes bin (column,row) from hDet map
  // and averages the gain in its (up to) 8 neighbours
  float average = 0.;
  std::vector<float> gainNeighbor;
  int offset = nbuffer / 2;

  for (int irow = 0; irow < nbuffer; irow++) {

    int rowNeighbor = (row - offset) + irow;
    if (rowNeighbor < 0 || rowNeighbor >= hDet->GetNbinsY()) {
      continue; // avoids under and overflow
    }
    for (int icol = 0; icol < nbuffer; icol++) {
      if (icol == 1 && irow == 1) {
        continue; // exclude self
      }
      int colNeighbor = (column - offset) + icol;
      if (colNeighbor < 0 || colNeighbor >= hDet->GetNbinsX()) {
        continue; // avoids under and overflow
      }
      float tempGain = TMath::Abs(hDet->GetBinContent(hDet->GetXaxis()->FindBin(colNeighbor), hDet->GetYaxis()->FindBin(rowNeighbor)));
      if (tempGain <= 0) {
        continue; // exclude empty/negative bins
      }
      gainNeighbor.push_back(tempGain);
    }
  }
  auto numberOfNeighbors = gainNeighbor.size();
  if (numberOfNeighbors < 1) {
    return 0; // if empty, return 0
  }

  average = std::accumulate(gainNeighbor.begin(), gainNeighbor.end(), decltype(gainNeighbor)::value_type(0.0)) / numberOfNeighbors;

  return average;
}

TH2F* PadCalibCCDBBuilder::getDetectorMap(TTree* tree, int nDet, float mingain, float maxgain, TString sDetName)
{ // creates a th2 map of a detector nDet
  // allows to limit the range of accepted adc gain by setting
  // mingain and maxgain (default range 0-10k)
  if (sDetName == "") {
    sDetName = Form("hDet%i", nDet);
  }
  int nTrdRows = NROWC1;
  int detInSupermodule = nDet % 30; // set the correct # of rows for stack 2
  if (detInSupermodule >= 12 && detInSupermodule <= 17) {
    nTrdRows = NROWC0;
  }
  // create the 2d map to be filled
  TH2F* hDetector = new TH2F(sDetName.Data(), sDetName.Data(), NCOLUMN, 0, NCOLUMN, nTrdRows, 0, nTrdRows);
  // set branches of our tree
  setTreeBranches(tree);
  // loop over tree and fill histo
  int nentries = tree->GetEntries();
  for (int ientry = 0; ientry < nentries; ientry++) {
    tree->GetEntry(ientry);
    if ((int)mDet != nDet) {
      continue;
    }
    if (mChi < 0 || mAmp <= 0 || mSgm <= 0 || mSgm > 1000) {
      continue;
    }
    // TODO add setters to change cuts
    //    if (mChiMax < mChiMin && mChi > mChiMax)
    //      continue;
    if (mAdc < mingain || mAdc > maxgain) {
      continue;
    }
    hDetector->SetBinContent(hDetector->GetXaxis()->FindBin(mCol), hDetector->GetYaxis()->FindBin(mRow), mAdc);
  }
  return hDetector;
}

bool PadCalibCCDBBuilder::isHotAreaIsolated(TH2F* hDet, int column, int row, int matrixSize)
{
  bool isIsolated = kFALSE;
  float averageGain = computeDetectorAverage(hDet);
  int nSurroundingNotHot = 0;
  int nMaxHot = TMath::Power(matrixSize + 2, 2) - TMath::Power(matrixSize, 2);
  float averageAround = 0.;
  int nUsedBins = 0;

  int nHotOffset = 0; // offsets the nMaxHot criterion for cells at edges
  if ((column == 1 || column == hDet->GetNbinsX()) && row > 1 && row < hDet->GetNbinsY()) {
    nHotOffset = matrixSize + 2;
  } else if ((row == 1 || row == hDet->GetNbinsY()) && column > 1 && column < hDet->GetNbinsX()) {
    nHotOffset = matrixSize + 2;
  } else if ((column == 1 || column == hDet->GetNbinsX()) && (row == 1 || row == hDet->GetNbinsY())) {
    nHotOffset = nMaxHot / 2 + 1;
  }

  for (int i = 0; i < matrixSize + 2; i++) {
    int icol = i - 1;
    for (int j = 0; j < matrixSize + 2; j++) {
      int jrow = j - 1;
      if ((-1 < icol) && (icol < matrixSize) && (-1 < jrow) && (jrow < matrixSize)) {
        continue;
      }
      float temp = TMath::Abs(hDet->GetBinContent(column + icol, row + jrow));
      if (temp != 0) {
        nUsedBins++;
        averageAround += temp;
      } else if (temp == 0) {
        nSurroundingNotHot++;
      }
    }
  }

  if (nUsedBins > 0) {
    averageAround /= nUsedBins;
  }
  if (averageAround < 2 * averageGain && nSurroundingNotHot <= nHotOffset) { //<= nMaxHot){
    isIsolated = kTRUE;
  } else if (nSurroundingNotHot == nMaxHot) {
    isIsolated = kTRUE;
  }

  return isIsolated;
}

int PadCalibCCDBBuilder::isolatedHotPadsContainmentSize(TH2F* hDet, int column, int row)
{
  int nsize = 0;
  bool isContained = kFALSE;
  while (!isContained) {
    nsize++;
    isContained = isHotAreaIsolated(hDet, column, row, nsize);
    if (nsize == 4) {
      return -1;
    }
  }
  return nsize;
}

void PadCalibCCDBBuilder::populateEmptyNormalizedMap(TH2F* hDet, float valueToSet)
{ // sets all cells in an empty normalized map to a given value
  // by default all cells set to -1

  if (!hDet || hDet->GetEntries() != 0) {
    std::cout << "Histogram does not exist or is not empty!";
    return;
  }
  for (int icol = 0; icol < hDet->GetNbinsX(); icol++) {
    for (int irow = 0; irow < hDet->GetNbinsY(); irow++) {
      hDet->SetBinContent(icol + 1, irow + 1, valueToSet);
    }
  }
}

void PadCalibCCDBBuilder::removeEdges(TH2F* hDet, int nsize)
{ // sets all cells in edges (along Y) of custom size (default size 2 columns) to 0
  if (!hDet || hDet->GetEntries() == 0) {
    return;
  }
  for (int icol = 0; icol < nsize; icol++) {
    for (int irow = 0; irow < hDet->GetNbinsY(); irow++) {
      hDet->SetBinContent(icol + 1, irow + 1, 0);
      hDet->SetBinContent(hDet->GetNbinsX() - icol, irow, 0);
    }
  }
}

void PadCalibCCDBBuilder::removeExtremePads(TH2F* hDet, float upperLimit, float lowerLimit)
{ // sets very hot (> 2*average) and very cold (< average/2) cells to 0
  if (!hDet || hDet->GetEntries() == 0) {
    return;
  }
  float average = computeDetectorAverage(hDet);

  for (int irow = 0; irow < hDet->GetNbinsY(); irow++) {
    for (int icol = 0; icol < hDet->GetNbinsX(); icol++) {
      float value = hDet->GetBinContent(icol + 1, irow + 1);
      if (value > upperLimit * average || value < lowerLimit * average) {
        hDet->SetBinContent(icol + 1, irow + 1, 0);
      }
    }
  }
}

void PadCalibCCDBBuilder::replacePadCloserToCenter(TH2F* hDet, int xcloser, int ycloser)
{
  if (!hDet || hDet->GetEntries() == 0) {
    std::cerr << "invalid histogram!" << std::endl;
    return;
  }

  float newGain = 0.;
  hDet->SetBinContent(xcloser, ycloser, newGain);
}

void PadCalibCCDBBuilder::replaceIsolatedHotPads(TH2F* hDet, int column, int row, int nsize)
{
  for (int jrow = 0; jrow < nsize; jrow++) {
    for (int icol = 0; icol < nsize; icol++) {
      hDet->SetBinContent(column + icol, row + jrow, 0);
    }
  }
}

void PadCalibCCDBBuilder::setTreeBranches(TTree* tree)
{
  tree->SetBranchAddress("det", &mDet);
  tree->SetBranchAddress("col", &mCol);
  tree->SetBranchAddress("row", &mRow);
  tree->SetBranchAddress("mean", &mAdc);
  tree->SetBranchAddress("chi2", &mChi);
  tree->SetBranchAddress("sigma", &mSgm);
  tree->SetBranchAddress("amplitude", &mAmp);
}

void PadCalibCCDBBuilder::smoothenTheDetector(TH2F* hDet, float allowedDifference)
{
  std::vector<std::vector<int>> SuspiciousBinPairs = findInhomogeneities(hDet, allowedDifference);
  auto numberOfPairs = SuspiciousBinPairs.size();
  if (numberOfPairs == 0) {
    return;
  }

  for (int i = 0; i < numberOfPairs; i++) {
    std::vector<int> susPair = SuspiciousBinPairs[i];
    checkIfIsolatedHotPadCandidate(hDet, susPair);
    checkIfSmallerCloserToCenter(hDet, susPair, allowedDifference);
  }
}

TH2F* PadCalibCCDBBuilder::transformMapIntoAbsoluteValues(TH2F* hDet, TString sName)
{
  if (!hDet || hDet->GetEntries() == 0) {
    return nullptr;
  }
  // set a name for the new histo
  if (sName == "") {
    sName = hDet->GetName();
    sName += "_transformed";
  }

  int nCols = hDet->GetNbinsX();
  int nRows = hDet->GetNbinsY();

  TH2F* hDetCopy = new TH2F(sName.Data(), sName.Data(), nCols, 0, nCols, nRows, 0, nRows);

  for (int icol = 0; icol < nCols; icol++) {
    for (int irow = 0; irow < nRows; irow++) {
      hDetCopy->SetBinContent(icol + 1, irow + 1, TMath::Abs(hDet->GetBinContent(icol + 1, irow + 1)));
    }
  }

  return hDetCopy;
}

} // namespace o2::trd
