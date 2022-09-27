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
#pragma link C++ nestedclasses;

#pragma link C++ class o2::tpc::AdcClockMonitor + ;
#pragma link C++ class o2::tpc::GBTFrame + ;
#pragma link C++ class o2::tpc::GBTFrameContainer + ;
#pragma link C++ class o2::tpc::HalfSAMPAData + ;
#pragma link C++ class o2::tpc::RawReader + ;
#pragma link C++ class o2::tpc::rawreader::RawReaderCRU + ;
#pragma link C++ class o2::tpc::rawreader::RawReaderCRUEventSync + ;
#pragma link C++ class o2::tpc::rawreader::RawReaderCRUManager + ;
#pragma link C++ class o2::tpc::RawReaderEventSync + ;
#pragma link C++ class o2::tpc::SyncPatternMonitor + ;
#pragma link C++ class o2::tpc::HardwareClusterDecoder + ;
#pragma link C++ class o2::tpc::KrBoxClusterFinder + ;

#pragma link C++ class o2::tpc::Clusterer + ;
#pragma link C++ class o2::tpc::ClustererTask + ;
#pragma link C++ class o2::tpc::HwClusterer + ;
#pragma link C++ class o2::tpc::HwClustererParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::HwClustererParam> + ;
#pragma link C++ class o2::tpc::KrBoxClusterFinderParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::KrBoxClusterFinderParam> + ;
#pragma link C++ class o2::tpc::IonTailCorrectionSettings + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::IonTailCorrectionSettings> + ;

#pragma link C++ class o2::tpc::TPCFastTransformHelperO2 + ;

#endif
