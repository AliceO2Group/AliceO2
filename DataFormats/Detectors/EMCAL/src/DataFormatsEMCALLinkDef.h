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

#pragma link C++ class o2::emcal::TriggerRecord + ;
#pragma link C++ class o2::dataformats::TimeStamp < Float16_t> + ;
#pragma link C++ class o2::emcal::Cell + ;
#pragma link C++ class o2::emcal::Digit + ;
#pragma link C++ class o2::emcal::Cluster + ;
#pragma link C++ class o2::emcal::AnalysisCluster + ;
#pragma link C++ class o2::emcal::MCLabel + ;
#pragma link C++ class o2::emcal::ErrorTypeFEE + ;

#pragma read \
  sourceClass = "o2::emcal::Cell" targetClass = "o2::emcal::Cell" source = "UShort_t mCellWords[3]" version = "[1]" include = "iostream" target = "mCellWords" code = "{const char *oldwords = reinterpret_cast<const char *>(onfile.mCellWords); for(int i = 0; i < 6; i++) {mCellWords[i] = oldwords[i];} }"

#pragma link C++ class std::vector < o2::emcal::TriggerRecord> + ;
#pragma link C++ class std::vector < o2::emcal::Cell> + ;
#pragma link C++ class std::vector < o2::emcal::Digit> + ;
#pragma link C++ class std::vector < o2::emcal::Cluster> + ;
#pragma link C++ class std::vector < o2::emcal::AnalysisCluster> + ;
#pragma link C++ class std::vector < o2::emcal::ErrorTypeFEE> + ;

#include "SimulationDataFormat/MCTruthContainer.h"
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::emcal::MCLabel> + ;

// For channel type in digits and cells
#pragma link C++ enum o2::emcal::ChannelType_t + ;

#pragma link C++ class std::vector < o2::emcal::Cluster> + ;

#pragma link C++ class o2::emcal::EventData < o2::emcal::Cell> + ;
#pragma link C++ class o2::emcal::EventData < o2::emcal::Digit> + ;
#pragma link C++ class o2::emcal::EventHandler < o2::emcal::Cell> + ;
#pragma link C++ class o2::emcal::EventHandler < o2::emcal::Digit> + ;

#pragma link C++ struct o2::emcal::CTFHeader + ;
#pragma link C++ struct o2::emcal::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::emcal::CTFHeader, 8, uint32_t> + ;

#endif
