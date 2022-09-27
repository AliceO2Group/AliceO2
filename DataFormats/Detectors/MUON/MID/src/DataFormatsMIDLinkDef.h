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

#pragma link C++ struct o2::mid::ChEffCounter + ;
#pragma link C++ class std::vector < o2::mid::ChEffCounter> + ;
#pragma link C++ struct o2::mid::Cluster + ;
#pragma link C++ class std::vector < o2::mid::Cluster> + ;
#pragma link C++ struct o2::mid::ColumnData + ;
#pragma link C++ class std::vector < o2::mid::ColumnData> + ;
#pragma link C++ struct o2::mid::ROFRecord + ;
#pragma link C++ class std::vector < o2::mid::ROFRecord> + ;
#pragma link C++ struct o2::mid::Track + ;
#pragma link C++ class std::vector < o2::mid::Track> + ;

#pragma link C++ struct o2::mid::CTFHeader + ;
#pragma link C++ struct o2::mid::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::mid::CTFHeader, 7, uint32_t> + ;

#include "SimulationDataFormat/MCTruthContainer.h"
#pragma link C++ class o2::mid::MCClusterLabel + ;
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::mid::MCClusterLabel> + ;
#pragma link C++ class o2::mid::MCLabel + ;
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::mid::MCLabel> + ;

#endif
