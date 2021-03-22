// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include "FITCalibration/FITViewObjectGenerator.h"

using namespace o2::calibration::fit;

template<typename ObjectToVisualize>
FITViewObjectGenerator::VisObjectsType
FITViewObjectGenerator::generateViewObjects(const ObjectToVisualize&)
{
  LOG(WARN) << "No viewer function for type : " << typeid(ObjectToVisualize).name() << "\n";
  return {};
}

//Add your specialization
template<>
FITViewObjectGenerator::VisObjectsType
FITViewObjectGenerator::generateViewObjects<FT0CalibrationObject>(const FT0CalibrationObject& obj)
{
  VisObjectsType visObjects;
  visObjects.emplace_back(FT0CalibrationObjectViewer::generateTGraphFromOffsetPoints(obj));
  return visObjects;
}

template<>
FITViewObjectGenerator::VisObjectsType
FITViewObjectGenerator::generateViewObjects<FT0ChannelDataTimeSlotContainer>(const FT0ChannelDataTimeSlotContainer& obj)
{
  VisObjectsType visObjects;
  visObjects.emplace_back(FT0ChannelDataTimeSlotContainerViewer::generateHistogramForValidChannels(obj));
  visObjects.emplace_back(FT0ChannelDataTimeSlotContainerViewer::generate2DHistogramTimeInFunctionOfChannel(obj));
  return visObjects;
}