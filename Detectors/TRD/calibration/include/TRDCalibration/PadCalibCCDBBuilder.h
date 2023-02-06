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

/// \file PadCalibCCDBBuilder.h
/// \brief Krypton calibration
/// \author Jana Crkovska

#ifndef O2_TRD_KRCALIBRATION_H
#define O2_TRD_KRCALIBRATION_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include "TH2.h"
#include "TTree.h"
#include <cstdlib>
#include <numeric>
#include <vector>

namespace o2
{
namespace trd
{

class PadCalibCCDBBuilder
{
 public:
  PadCalibCCDBBuilder() = default;
  ~PadCalibCCDBBuilder() = default;

  void checkIfIsolatedHotPadCandidate(TH2F* hDet, std::vector<int> coordinates, float upperLimit = 1.5, int areaContainedWithin = 4);
  void checkIfSmallerCloserToCenter(TH2F* hDet, std::vector<int> coordinates, float allowedDifference);
  std::vector<int> compareGain(TH2F* hDet, int column, int row, int shiftcolumn, int shiftrow, float allowedDifference);
  float computeDetectorAverage(TH2F* hDet);
  float computeDistance(std::vector<float> pad1, std::vector<float> pad2);
  TH2F* createNormalizedMap(TH2F* hDet, TString sNewName = "");
  void fillInTheGap(TH2F* hDet, int column, int row, float newGain);
  TH2F* fillTheMap(TH2F* hDet, TString sNewName = "", int nbuffer = 3);
  std::vector<std::vector<int>> findEmpty(TH2F* hDetectorMap);
  std::vector<std::vector<int>> findInhomogeneities(TH2F* hDet, float allowedDifference);
  float getAverageFromNeighbors(TH2F* hDet, int column, int row, int nbuffer = 3);
  TH2F* getDetectorMap(TTree* tree, int nDet, float mingain = 0, float maxgain = 10'000, TString sDetName = "");
  bool isHotAreaIsolated(TH2F* hDet, int column, int row, int matrixSize = 1);
  int isolatedHotPadsContainmentSize(TH2F* hDet, int column, int row);
  void populateEmptyNormalizedMap(TH2F* hDet, float valueToSet = -1);
  void removeEdges(TH2F* hDet, int nsize = 2);
  void removeExtremePads(TH2F* hDet, float upperLimit = 2., float lowerLimit = 0.5);
  void replacePadCloserToCenter(TH2F* hDet, int column, int row);
  void replaceIsolatedHotPads(TH2F* hDet, int column, int row, int nsize);
  void setTreeBranches(TTree* tree);
  void smoothenTheDetector(TH2F* hDet, float allowedDifference = 1000);
  TH2F* transformMapIntoAbsoluteValues(TH2F* hDet, TString sName = "");

 private:
  float mDet;
  float mCol;
  float mRow;
  float mAdc;
  float mChi;
  float mSgm;
  float mAmp;

  ClassDefNV(PadCalibCCDBBuilder, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_KRCALIBRATION_H