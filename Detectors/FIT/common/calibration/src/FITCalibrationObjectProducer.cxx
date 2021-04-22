// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITCalibration/FITCalibrationObjectProducer.h"

using namespace o2::fit;

//One can extract needed CalibrationObject from FITCalibrationApi and send to proper function from lower module

template <typename CalibrationObjectType, typename TimeSlotContainerType>
CalibrationObjectType FITCalibrationObjectProducer::generateCalibrationObject(const TimeSlotContainerType& container)
{
  throw std::runtime_error("Cannot find proper overload for provided calibration object type");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
o2::ft0::FT0ChannelTimeCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::ft0::FT0ChannelTimeCalibrationObject, o2::ft0::FT0ChannelTimeTimeSlotContainer>(const o2::ft0::FT0ChannelTimeTimeSlotContainer& container)
{
  return o2::ft0::FT0TimeChannelOffsetCalibrationObjectAlgorithm::generateCalibrationObject(container);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
