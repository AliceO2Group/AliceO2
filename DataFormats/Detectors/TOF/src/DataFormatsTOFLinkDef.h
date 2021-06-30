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
#pragma link C++ class o2::tof::Cluster + ;
#pragma link C++ class o2::tof::CalibInfoCluster + ;
#pragma link C++ class o2::tof::CosmicInfo + ;
#pragma link C++ class o2::tof::CalibInfoTrackCl + ;
#pragma link C++ class std::vector < o2::tof::Cluster> + ;
#pragma link C++ class std::vector < o2::tof::CalibInfoCluster> + ;
#pragma link C++ class std::vector < o2::tof::CosmicInfo> + ;
#pragma link C++ class std::vector < o2::tof::CalibInfoTrackCl> + ;

#pragma link C++ class o2::dataformats::CalibInfoTOFshort + ;
#pragma link C++ class o2::dataformats::CalibInfoTOF + ;

#pragma link C++ class o2::dataformats::CalibLHCphaseTOF + ;
#pragma link C++ class o2::dataformats::CalibTimeSlewingParamTOF + ;

#pragma link C++ class std::vector < o2::dataformats::CalibInfoTOFshort> + ;
#pragma link C++ class std::vector < o2::dataformats::CalibInfoTOF> + ;

#pragma link C++ class o2::tof::CTFHeader + ;
#pragma link C++ class o2::tof::CompressedInfos + ;
#pragma link C++ class o2::tof::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::tof::CTFHeader, 11, uint32_t> + ;

#endif
