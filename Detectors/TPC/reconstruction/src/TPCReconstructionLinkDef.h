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

#pragma link C++ class o2::TPC::AdcClockMonitor;
#pragma link C++ class o2::TPC::GBTFrame;
#pragma link C++ class o2::TPC::GBTFrameContainer;
#pragma link C++ class o2::TPC::HalfSAMPAData;
#pragma link C++ class o2::TPC::RawReader;
#pragma link C++ class o2::TPC::RawReaderEventSync;
#pragma link C++ class o2::TPC::SyncPatternMonitor;
#pragma link C++ class o2::TPC::TrackTPC+;
#pragma link C++ class o2::TPC::TPCCATracking;
#pragma link C++ class o2::TPC::HardwareClusterDecoder;

#pragma link C++ class o2::TPC::BoxClusterer+;
#pragma link C++ class o2::TPC::ClusterTimeStamp+;
#pragma link C++ class o2::TPC::Cluster+;
#pragma link C++ class o2::TPC::Clusterer+;
#pragma link C++ class o2::TPC::ClusterContainer+;
#pragma link C++ class o2::TPC::ClustererTask+;
#pragma link C++ class o2::TPC::HwClusterer+;
#pragma link C++ class o2::TPC::HwClusterFinder+;
#pragma link C++ class o2::TPC::HwFixedPoint+;

#pragma link C++ class std::vector<o2::TPC::TrackTPC>+;

#endif
