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

#pragma link C++ class o2::tpc::AdcClockMonitor;
#pragma link C++ class o2::tpc::GBTFrame;
#pragma link C++ class o2::tpc::GBTFrameContainer;
#pragma link C++ class o2::tpc::HalfSAMPAData;
#pragma link C++ class o2::tpc::RawReader;
#pragma link C++ class o2::tpc::RawReaderCRU;
#pragma link C++ class o2::tpc::RawReaderEventSync;
#pragma link C++ class o2::tpc::SyncPatternMonitor;
#pragma link C++ class o2::tpc::GPUCATracking;
#pragma link C++ class o2::tpc::HardwareClusterDecoder;

#pragma link C++ class o2::tpc::Clusterer + ;
#pragma link C++ class o2::tpc::ClustererTask + ;
#pragma link C++ class o2::tpc::HwClusterer + ;

#pragma link C++ class o2::tpc::TPCFastTransformHelperO2 + ;

#endif
