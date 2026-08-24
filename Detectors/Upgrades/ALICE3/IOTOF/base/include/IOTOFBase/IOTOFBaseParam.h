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

#ifndef O2_IOTOF_BASEPARAM_H
#define O2_IOTOF_BASEPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace iotof
{

struct ChipSpecifics {
  int NCols = 129;
  int NRows = 271;
  float PitchCol = 250.00e-4;
  float PitchRow = 100.00e-4;
  float PassiveEdgeReadOut = 0.;
  float PassiveEdgeTop = 0.;
  float PassiveEdgeSide = 0.;
  float PixelPassiveEdgeX = 0.;
  float PixelPassiveEdgeZ = 0.;
  float SensorLayerThicknessEff = 50.e-4;
  float SensorLayerThickness = 50.e-4;

  int NPixels() const { return NCols * NRows; }
  float ActiveMatrixSizeCols() const { return PitchCol * NCols; }
  float ActiveMatrixSizeRows() const { return PitchRow * NRows; }
  float SensorSizeCols() const { return ActiveMatrixSizeCols() + PassiveEdgeSide + PassiveEdgeSide; }
  float SensorSizeRows() const { return ActiveMatrixSizeRows() + PassiveEdgeTop + PassiveEdgeReadOut; }
};

struct ChipSpecificsParam : public o2::conf::ConfigurableParamPromoter<ChipSpecificsParam, ChipSpecifics> {
  O2ParamDef(ChipSpecificsParam, "ChipSpecificsParam")
};

struct IOTOFBaseParam : public o2::conf::ConfigurableParamHelper<IOTOFBaseParam> {
  bool enableInnerTOF = true;       // Enable Inner TOF layer
  bool enableOuterTOF = true;       // Enable Outer TOF layer
  bool enableForwardTOF = true;     // Enable Forward TOF layer
  bool enableBackwardTOF = true;    // Enable Backward TOF layer
  std::string detectorPattern = ""; // Layouts of the detector
  bool segmentedInnerTOF = false;   // If the inner TOF layer is segmented
  bool segmentedOuterTOF = false;   // If the outer TOF layer is segmented
  float x2x0 = 0.02f;               // thickness expressed in radiation length, for all layers for the moment
  float sensorThickness = 0.0050f;  // thickness of the sensor in cm, for all layers for the moment, the default is set to 50 microns

  O2ParamDef(IOTOFBaseParam, "IOTOFBase");
};

} // namespace iotof
} // end namespace o2

#endif
