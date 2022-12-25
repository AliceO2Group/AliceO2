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

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::ft0::FT0TimeOffsetSlotContainer + ;
#pragma link C++ class o2::ft0::GlobalOffsetsContainer + ;
#pragma link C++ class o2::ft0::FT0CalibTimeSlewing + ;
#pragma link C++ class o2::ft0::FT0CalibCollector + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::ft0::FT0CalibInfoSlot>;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::ft0::FT0CalibInfoSlot>;
#pragma link C++ class o2::calibration::TimeSlot < o2::ft0::FT0TimeOffsetSlotContainer>;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::ft0::FT0TimeOffsetSlotContainer>;
#pragma link C++ class o2::calibration::TimeSlot < o2::ft0::GlobalOffsetsContainer>;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::ft0::GlobalOffsetsContainer>;

#endif
