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

#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "DataFormatsFT0/GlobalOffsetsContainer.h"
#include "Framework/Logger.h"
#include <TString.h>
using namespace o2::ft0;

GlobalOffsetsCalibrationObject GlobalOffsetsCalibrationObjectAlgorithm::generateCalibrationObject(const GlobalOffsetsContainer& container)
{
  GlobalOffsetsCalibrationObject calibrationObject;
  calibrationObject.mCollisionTimeOffsets = container.getMeanGaussianFitValue();
  LOG(info) << "GlobalOffsetsCalibrationObjectAlgorithm generate CalibrationObject for T0"
             << " = " << calibrationObject.mCollisionTimeOffsets;

  return calibrationObject;
}
