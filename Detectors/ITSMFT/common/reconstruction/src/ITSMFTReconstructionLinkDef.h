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

#pragma link C++ class o2::itsmft::Clusterer + ;
#pragma link C++ class o2::itsmft::PixelReader + ;
#pragma link C++ class o2::itsmft::DigitPixelReader + ;
#pragma link C++ class o2::itsmft::RawPixelReader < o2::itsmft::ChipMappingITS> + ;
#pragma link C++ class o2::itsmft::RawPixelReader < o2::itsmft::ChipMappingMFT> + ;
#pragma link C++ class o2::itsmft::PixelData + ;
#pragma link C++ class o2::itsmft::ChipPixelData + ;
#pragma link C++ class o2::itsmft::BuildTopologyDictionary + ;
#pragma link C++ class o2::itsmft::LookUp + ;
#pragma link C++ class o2::itsmft::TopologyFastSimulation + ;
#pragma link C++ class o2::itsmft::ChipMappingITS + ;
#pragma link C++ class o2::itsmft::ChipMappingMFT + ;
#pragma link C++ class o2::itsmft::AlpideCoder + ;
#pragma link C++ class o2::itsmft::GBTWord + ;
#pragma link C++ class o2::itsmft::GBTDataHeader + ;
#pragma link C++ class o2::itsmft::GBTDataTrailer + ;
#pragma link C++ class o2::itsmft::GBTData + ;
#pragma link C++ class o2::itsmft::PayLoadCont + ;
#pragma link C++ class o2::itsmft::PayLoadSG + ;
#pragma link C++ class o2::itsmft::GBTLinkDecodingStat + ;
#pragma link C++ class o2::itsmft::RawDecodingStat + ;

#pragma link C++ class std::map < unsigned long, std::pair < o2::itsmft::ClusterTopology, unsigned long>> + ;

#endif
