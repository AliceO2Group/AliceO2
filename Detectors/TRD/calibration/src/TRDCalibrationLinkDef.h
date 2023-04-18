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

#pragma link C++ class o2::trd::CalibratorVdExB + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::trd::AngularResidHistos> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::trd::AngularResidHistos> + ;
#pragma link C++ class o2::trd::CalibratorNoise + ;
#pragma link C++ class o2::trd::ChannelInfoDetailed + ;
#pragma link C++ class o2::trd::TrackBasedCalib + ;
#pragma link C++ class o2::trd::PadCalibCCDBBuilder + ;
#pragma link C++ class o2::trd::KrClusterFinder + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, o2::trd::TRDDCSMinMaxMeanInfo> + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, float> + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, int> + ;
#pragma link C++ class o2::trd::TRDCalibParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::trd::TRDCalibParams> + ;

#endif
