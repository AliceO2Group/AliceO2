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

Segmentation::Segmentation()
{
  auto& iotofPars = IOTOFBaseParam::Instance();
  const ChipSpecifics& iTofChipPars = iotofPars.iTofChipSpecifics;
  const ChipSpecifics& oTofChipPars = iotofPars.oTofChipSpecifics;
  
  configChip(iTofChipPars, 0 /* subDetectorID for iTOF */);
  configChip(oTofChipPars, 1 /* subDetectorID for oTOF */);
}

void Segmentation::configChip(const int nCols, const int nRows, const float pitchCol, const float pitchRow, const float passiveEdgeReadOut,
                          const float passiveEdgeTop, const float passiveEdgeSide, const float sensorLayerThicknessEff, const float sensorLayerThickness, const int subDetectorID)
{
  if (subDetectorID == 0) {
    iTofSpecsConfig = ChipSpecifics(nCols, nRows, pitchCol, pitchRow, passiveEdgeReadOut, passiveEdgeTop, passiveEdgeSide, sensorLayerThicknessEff, sensorLayerThickness);
  } else if (subDetectorID == 1) {
    oTofSpecsConfig = ChipSpecifics(nCols, nRows, pitchCol, pitchRow, passiveEdgeReadOut, passiveEdgeTop, passiveEdgeSide, sensorLayerThicknessEff, sensorLayerThickness);
  } else {
    printf("Invalid subDetectorID %d. Must be 0 (iTOF) or 1 (oTOF). No configuration applied.\n", subDetectorID);
  }
}

void Segmentation::configChip(const ChipSpecifics& specsConfig, const int subDetectorID)
{
  if (subDetectorID == 0) {
    iTofSpecsConfig = specsConfig;
  } else if (subDetectorID == 1) {
    oTofSpecsConfig = specsConfig;
  } else {
    printf("Invalid subDetectorID %d. Must be 0 (iTOF) or 1 (oTOF). No configuration applied.\n", subDetectorID);
  }
}

void Segmentation::print()
{
  // iTOF specs
  printf("iTOF specs:\n");
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", iTofSpecsConfig.PitchRow * 1e4, iTofSpecsConfig.NRows, iTofSpecsConfig.PitchCol * 1e4, iTofSpecsConfig.NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n", iTofSpecsConfig.PassiveEdgeReadOut * 1e4, iTofSpecsConfig.PassiveEdgeTop * 1e4, iTofSpecsConfig.PassiveEdgeSide * 1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n", iTofSpecsConfig.ActiveMatrixSizeRows(), iTofSpecsConfig.SensorSizeRows(), iTofSpecsConfig.ActiveMatrixSizeCols(), iTofSpecsConfig.SensorSizeCols());

  // oTOF specs
  printf("oTOF specs:\n");
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", oTofSpecsConfig.PitchRow * 1e4, oTofSpecsConfig.NRows, oTofSpecsConfig.PitchCol * 1e4, oTofSpecsConfig.NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n", oTofSpecsConfig.PassiveEdgeReadOut * 1e4, oTofSpecsConfig.PassiveEdgeTop * 1e4, oTofSpecsConfig.PassiveEdgeSide * 1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n", oTofSpecsConfig.ActiveMatrixSizeRows(), oTofSpecsConfig.SensorSizeRows(), oTofSpecsConfig.ActiveMatrixSizeCols(), oTofSpecsConfig.SensorSizeCols());
}

} // namespace iotof
} // namespace o2

ClassImp(o2::iotof::Segmentation);
