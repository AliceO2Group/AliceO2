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

/// \file Segmentation.cxx
/// \brief Implementation of the Segmentation class

#include "IOTOFSimulation/Segmentation.h"
#include "IOTOFBase/IOTOFBaseParam.h"
#include <cstdio>

namespace o2
{

namespace iotof
{

std::unique_ptr<o2::iotof::Segmentation> Segmentation::sInstance;

Segmentation* Segmentation::Instance()
{
  if (!sInstance) {
    sInstance = std::unique_ptr<Segmentation>(new Segmentation());
  }
  return sInstance.get();
}

Segmentation::Segmentation()
{
  if (sInstance) {
    printf("Invalid use of public constructor: o2::iotof::Segmentation instance exists\n");
  } else {
    auto& itofPars = ITOFChipSpecificParam::Instance();
    auto& otofPars = OTOFChipSpecificParam::Instance();
    const ChipSpecifics mITofChipPars(itofPars.NCols, itofPars.NRows, itofPars.PitchCol, itofPars.PitchRow, itofPars.PassiveEdgeReadOut, itofPars.PassiveEdgeTop, itofPars.PassiveEdgeSide, itofPars.SensorLayerThicknessEff, itofPars.SensorLayerThickness);
    const ChipSpecifics mOTofChipPars(otofPars.NCols, otofPars.NRows, otofPars.PitchCol, otofPars.PitchRow, otofPars.PassiveEdgeReadOut, otofPars.PassiveEdgeTop, otofPars.PassiveEdgeSide, otofPars.SensorLayerThicknessEff, otofPars.SensorLayerThickness);

    configChip(mITofChipPars, 0 /* subDetectorID for iTOF */);
    configChip(mOTofChipPars, 1 /* subDetectorID for oTOF */);
  }
}

void Segmentation::configChip(const int nCols, const int nRows, const float pitchCol, const float pitchRow, const float passiveEdgeReadOut,
                              const float passiveEdgeTop, const float passiveEdgeSide, const float sensorLayerThicknessEff, const float sensorLayerThickness, const int subDetectorID)
{
  if (subDetectorID == 0) {
    mITofSpecsConfig = ChipSpecifics(nCols, nRows, pitchCol, pitchRow, passiveEdgeReadOut, passiveEdgeTop, passiveEdgeSide, sensorLayerThicknessEff, sensorLayerThickness);
  } else if (subDetectorID == 1) {
    mOTofSpecsConfig = ChipSpecifics(nCols, nRows, pitchCol, pitchRow, passiveEdgeReadOut, passiveEdgeTop, passiveEdgeSide, sensorLayerThicknessEff, sensorLayerThickness);
  } else {
    printf("Invalid subDetectorID %d. Must be 0 (iTOF) or 1 (oTOF). No configuration applied.\n", subDetectorID);
  }
}

void Segmentation::configChip(const ChipSpecifics& specsConfig, const int subDetectorID)
{
  if (subDetectorID == 0) {
    mITofSpecsConfig = specsConfig;
  } else if (subDetectorID == 1) {
    mOTofSpecsConfig = specsConfig;
  } else {
    printf("Invalid subDetectorID %d. Must be 0 (iTOF) or 1 (oTOF). No configuration applied.\n", subDetectorID);
  }
}

void Segmentation::print()
{
  // iTOF specs
  printf("iTOF specs:\n");
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", mITofSpecsConfig.PitchRow * 1e4, mITofSpecsConfig.NRows, mITofSpecsConfig.PitchCol * 1e4, mITofSpecsConfig.NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n", mITofSpecsConfig.PassiveEdgeReadOut * 1e4, mITofSpecsConfig.PassiveEdgeTop * 1e4, mITofSpecsConfig.PassiveEdgeSide * 1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n", mITofSpecsConfig.ActiveMatrixSizeRows(), mITofSpecsConfig.SensorSizeRows(), mITofSpecsConfig.ActiveMatrixSizeCols(), mITofSpecsConfig.SensorSizeCols());

  // oTOF specs
  printf("oTOF specs:\n");
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", mOTofSpecsConfig.PitchRow * 1e4, mOTofSpecsConfig.NRows, mOTofSpecsConfig.PitchCol * 1e4, mOTofSpecsConfig.NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n", mOTofSpecsConfig.PassiveEdgeReadOut * 1e4, mOTofSpecsConfig.PassiveEdgeTop * 1e4, mOTofSpecsConfig.PassiveEdgeSide * 1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n", mOTofSpecsConfig.ActiveMatrixSizeRows(), mOTofSpecsConfig.SensorSizeRows(), mOTofSpecsConfig.ActiveMatrixSizeCols(), mOTofSpecsConfig.SensorSizeCols());
}

} // namespace iotof
} // namespace o2

ClassImp(o2::iotof::Segmentation);
