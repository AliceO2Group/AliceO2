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

int Segmentation::NCols = 0;
int Segmentation::NRows = 0;
int Segmentation::NPixels = 0;
float Segmentation::PitchCol = 0.f;
float Segmentation::PitchRow = 0.f;
float Segmentation::PassiveEdgeReadOut = 0.f;
float Segmentation::PassiveEdgeTop = 0.f;
float Segmentation::PassiveEdgeSide = 0.f;
float Segmentation::ActiveMatrixSizeCols = 0.f;
float Segmentation::ActiveMatrixSizeRows = 0.f;
float Segmentation::SensorLayerThicknessEff = 0.f;
float Segmentation::SensorLayerThickness = 0.f;
float Segmentation::SensorSizeCols = 0.f;
float Segmentation::SensorSizeRows = 0.f;

Segmentation::Segmentation()
{
  auto& iotofPars = IOTOFBaseParam::Instance();
  auto& chipPars = iotofPars.chipSpecifics;
  configChip(chipPars.NCols, chipPars.NRows, chipPars.PitchCol, chipPars.PitchRow, chipPars.PassiveEdgeReadOut, chipPars.PassiveEdgeTop,
             chipPars.PassiveEdgeSide, chipPars.SensorLayerThicknessEff, chipPars.SensorLayerThickness);
}

void Segmentation::configChip(const int nCols, const int nRows, const float pitchCol, const float pitchRow, const float passiveEdgeReadOut,
                          const float passiveEdgeTop, const float passiveEdgeSide, const float sensorLayerThicknessEff, const float sensorLayerThickness)
{
  NCols = nCols;
  NRows = nRows;
  NPixels = NCols * NRows;
  PitchCol = pitchCol;
  PitchRow = pitchRow;
  PassiveEdgeReadOut = passiveEdgeReadOut;
  PassiveEdgeTop = passiveEdgeTop;
  PassiveEdgeSide = passiveEdgeSide;
  ActiveMatrixSizeCols = PitchCol * NCols;
  ActiveMatrixSizeRows = PitchRow * NRows;
  SensorLayerThicknessEff = sensorLayerThicknessEff;
  SensorLayerThickness = sensorLayerThickness;
  SensorSizeCols = ActiveMatrixSizeCols + PassiveEdgeSide + PassiveEdgeSide;
  SensorSizeRows = ActiveMatrixSizeRows + PassiveEdgeTop + PassiveEdgeReadOut;
}

void Segmentation::print()
{
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", PitchRow * 1e4, NRows, PitchCol * 1e4, NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n",
         PassiveEdgeReadOut * 1e4, PassiveEdgeTop * 1e4, PassiveEdgeSide * 1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n", ActiveMatrixSizeRows, SensorSizeRows,
         ActiveMatrixSizeCols, SensorSizeCols);
}

} // namespace iotof
} // namespace o2

ClassImp(o2::iotof::Segmentation);
