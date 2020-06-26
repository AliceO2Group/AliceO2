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

#pragma link C++ class o2::endcaps::RawPixelDecoder < o2::endcaps::ChipMappingITS> + ;
#pragma link C++ class o2::endcaps::RawPixelDecoder < o2::endcaps::ChipMappingMFT> + ;

#pragma link C++ class o2::endcaps::Clusterer + ;
#pragma link C++ class o2::endcaps::PixelReader + ;
#pragma link C++ class o2::endcaps::DigitPixelReader + ;
#pragma link C++ class o2::endcaps::RawPixelReader < o2::endcaps::ChipMappingITS> + ;
#pragma link C++ class o2::endcaps::RawPixelReader < o2::endcaps::ChipMappingMFT> + ;

#pragma link C++ class o2::endcaps::PixelData + ;
#pragma link C++ class o2::endcaps::ChipPixelData + ;
#pragma link C++ class o2::endcaps::BuildTopologyDictionary + ;
#pragma link C++ class o2::endcaps::LookUp + ;
#pragma link C++ class o2::endcaps::TopologyFastSimulation + ;
#pragma link C++ class o2::endcaps::ChipMappingITS + ;
#pragma link C++ class o2::endcaps::ChipMappingMFT + ;
#pragma link C++ class o2::endcaps::AlpideCoder + ;
#pragma link C++ class o2::endcaps::GBTWord + ;
#pragma link C++ class o2::endcaps::GBTDataHeader + ;
#pragma link C++ class o2::endcaps::GBTDataTrailer + ;
#pragma link C++ class o2::endcaps::GBTData + ;
#pragma link C++ class o2::endcaps::PayLoadCont + ;
#pragma link C++ class o2::endcaps::PayLoadSG + ;
#pragma link C++ class o2::endcaps::GBTLinkDecodingStat + ;
#pragma link C++ class o2::endcaps::GBTLink + ;
#pragma link C++ class o2::endcaps::RUDecodeData + ;
#pragma link C++ class o2::endcaps::RawDecodingStat + ;

#pragma link C++ class std::map < unsigned long, std::pair < o2::itsmft::ClusterTopology, unsigned long>> + ;

#pragma link C++ class o2::endcaps::ClustererParam < o2::detectors::DetID::EC0> + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::endcaps::ClustererParam < o2::detectors::DetID::EC0>> + ;

#endif
