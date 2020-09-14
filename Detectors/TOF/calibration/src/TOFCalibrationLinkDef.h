// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::tof::CalibTOFapi + ;
#pragma link C++ class o2::globaltracking::CalibTOF + ;
#pragma link C++ class o2::globaltracking::CollectCalibInfoTOF + ;

#pragma link C++ class o2::tof::LHCClockDataHisto + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tof::LHCClockDataHisto> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::dataformats::CalibInfoTOF, o2::tof::LHCClockDataHisto> + ;
#pragma link C++ class o2::tof::LHCClockCalibrator + ;

#pragma link C++ class o2::tof::TOFChannelData + ;
#pragma link C++ class o2::tof::TOFChannelCalibrator + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tof::TOFChannelData> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::dataformats::CalibInfoTOF, o2::tof::TOFChannelData> + ;

#pragma link C++ class o2::tof::TOFCalibInfoSlot + ;
#pragma link C++ class o2::tof::TOFCalibCollector + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tof::TOFCalibInfoSlot> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::dataformats::CalibInfoTOF, o2::tof::TOFCalibInfoSlot> + ;

#endif
