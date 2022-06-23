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

#pragma link C++ class o2::globaltracking::CalibTOF + ;
#pragma link C++ class o2::globaltracking::CollectCalibInfoTOF + ;

#pragma link C++ class o2::tof::LHCClockDataHisto + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tof::LHCClockDataHisto> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::dataformats::CalibInfoTOF, o2::tof::LHCClockDataHisto> + ;
#pragma link C++ class o2::tof::LHCClockCalibrator + ;

#pragma link C++ class o2::tof::TOFChannelData + ;
#pragma link C++ class o2::tof::TOFChannelCalibrator < o2::dataformats::CalibInfoTOF> + ;
#pragma link C++ class o2::tof::TOFChannelCalibrator < o2::tof::CalibInfoCluster> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tof::TOFChannelData> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::dataformats::CalibInfoTOF, o2::tof::TOFChannelData> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tof::CalibInfoCluster, o2::tof::TOFChannelData> + ;

#pragma link C++ class o2::tof::TOFCalibInfoSlot + ;
#pragma link C++ class o2::tof::TOFCalibCollector + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tof::TOFCalibInfoSlot> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::dataformats::CalibInfoTOF, o2::tof::TOFCalibInfoSlot> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tof::CalibInfoCluster, o2::tof::TOFCalibInfoSlot> + ;

#pragma link C++ class std::bitset < o2::tof::Geo::NCHANNELS> + ;
#pragma link C++ struct std::pair < uint64_t, double> + ;
#pragma link C++ struct std::pair < std::array < uint64_t, 2>, double> + ;
#pragma link C++ struct o2::tof::TOFDCSinfo + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, o2::tof::TOFDCSinfo> + ;

#pragma link C++ struct TOFFEEchannelConfig + ;
#pragma link C++ struct TOFFEEtriggerConfig + ;
#pragma link C++ struct TOFFEElightConfig + ;
#pragma link C++ struct TOFFEElightReader + ;

#pragma link C++ class o2::calibration::TimeSlot < o2::tof::Diagnostic> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tof::Diagnostic, o2::tof::Diagnostic> + ;
#pragma link C++ class o2::tof::TOFDiagnosticCalibrator + ;

#endif
