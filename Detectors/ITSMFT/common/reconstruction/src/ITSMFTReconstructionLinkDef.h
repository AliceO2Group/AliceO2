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

#pragma link C++ class o2::ITSMFT::Clusterer + ;
#pragma link C++ class o2::ITSMFT::PixelReader + ;
#pragma link C++ class o2::ITSMFT::DigitPixelReader + ;
#pragma link C++ class o2::ITSMFT::RawPixelReader < o2::ITSMFT::ChipMappingITS > +;
#pragma link C++ class o2::ITSMFT::RawPixelReader < o2::ITSMFT::ChipMappingMFT > +;
#pragma link C++ class o2::ITSMFT::PixelData + ;
#pragma link C++ class o2::ITSMFT::ChipPixelData + ;
#pragma link C++ class o2::ITSMFT::BuildTopologyDictionary + ;
#pragma link C++ class o2::ITSMFT::LookUp + ;
#pragma link C++ class o2::ITSMFT::TopologyFastSimulation + ;
#pragma link C++ struct o2::ITSMFT::ITSChipMappingData + ;
#pragma link C++ struct o2::ITSMFT::ITSModuleMappingData + ;
#pragma link C++ class o2::ITSMFT::ChipMappingITS + ;
#pragma link C++ class o2::ITSMFT::ChipMappingMFT + ;
#pragma link C++ class o2::ITSMFT::AlpideCoder + ;

#pragma link C++ class std::map<unsigned long, std::pair<o2::ITSMFT::ClusterTopology, unsigned long>> + ;



#endif
