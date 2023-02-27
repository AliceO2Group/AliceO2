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

#pragma link C++ class o2::mch::ROFRecord + ;
#pragma link C++ class o2::mch::TrackMCH + ;
#pragma link C++ class std::vector < o2::mch::TrackMCH> + ;
#pragma link C++ class o2::mch::Cluster + ;
#pragma link C++ class std::vector < o2::mch::Cluster> + ;
#pragma link C++ class o2::mch::DsChannelId + ;
#pragma link C++ class o2::mch::DsChannelGroup + ;
#pragma link C++ class std::vector < o2::mch::DsChannelId> + ;
#pragma link C++ class o2::mch::DsChannelDetId + ;
#pragma link C++ class std::vector < o2::mch::DsChannelDetId> + ;
#pragma link C++ class o2::mch::Digit + ;
#pragma link C++ class std::vector < o2::mch::Digit> + ;
#pragma link C++ struct o2::mch::CTFHeader + ;
#pragma link C++ struct o2::mch::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::mch::CTFHeader, 9, uint32_t> + ;
#pragma link C++ class std::vector < o2::mch::ROFRecord> + ;

#endif
