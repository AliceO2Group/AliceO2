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

#pragma link C++ class o2::emcal::CalibDB + ;
#pragma link C++ class o2::emcal::BadChannelMap + ;
#pragma link C++ class o2::emcal::TimeCalibrationParams + ;
#pragma link C++ class o2::emcal::TimeCalibParamL1Phase + ;
#pragma link C++ class o2::emcal::TempCalibrationParams + ;
#pragma link C++ class o2::emcal::TempCalibParamSM + ;
#pragma link C++ class o2::emcal::GainCalibrationFactors + ;
#pragma link C++ class o2::emcal::TriggerTRUDCS + ;
#pragma link C++ class o2::emcal::TriggerSTUDCS + ;
#pragma link C++ class o2::emcal::TriggerSTUErrorCounter + ;
#pragma link C++ class o2::emcal::TriggerDCS + ;
#pragma link C++ class o2::emcal::FeeDCS + ;
#pragma link C++ class o2::emcal::ElmbData + ;
#pragma link C++ class o2::emcal::ElmbMeasurement + ;
#pragma link C++ class o2::emcal::EMCALChannelScaleFactors + ;
#pragma link C++ class o2::emcal::EnergyIntervals + ;
#pragma link C++ class std::map < o2::emcal::EnergyIntervals, float> + ;

#endif
