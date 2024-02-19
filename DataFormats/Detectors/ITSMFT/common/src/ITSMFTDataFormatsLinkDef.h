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

#pragma link C++ class o2::itsmft::Digit + ;
#pragma link C++ class o2::itsmft::NoiseMap + ;
#pragma link C++ class o2::itsmft::TimeDeadMap + ;
#pragma link C++ class std::vector < o2::itsmft::Digit> + ;

#pragma link C++ class o2::itsmft::GBTCalibData + ;
#pragma link C++ class std::vector < o2::itsmft::GBTCalibData> + ;

#pragma link C++ class o2::itsmft::ROFRecord + ;
#pragma link C++ class std::vector < o2::itsmft::ROFRecord> + ;
#pragma link C++ class o2::itsmft::MC2ROFRecord + ;
#pragma link C++ class std::vector < o2::itsmft::MC2ROFRecord> + ;
#pragma link C++ class o2::itsmft::Cluster + ;
#pragma link C++ class o2::itsmft::CompCluster + ;
#pragma link C++ class o2::itsmft::CompClusterExt + ;
#pragma link C++ class std::vector < o2::itsmft::Cluster> + ;
#pragma link C++ class std::vector < o2::itsmft::CompCluster> + ;
#pragma link C++ class std::vector < o2::itsmft::CompClusterExt> + ;
#pragma link C++ class o2::itsmft::ClusterPattern + ;
#pragma link C++ class std::map < int, o2::itsmft::ClusterPattern> + ;
#pragma link C++ class o2::itsmft::ClusterTopology + ;
#pragma link C++ class o2::itsmft::TopologyDictionary + ;
#pragma link C++ class o2::itsmft::GroupStruct + ;

#pragma link C++ class o2::itsmft::TrkClusRef + ;
#pragma link C++ class std::vector < o2::itsmft::TrkClusRef> + ;

#pragma link C++ class o2::itsmft::PhysTrigger + ;
#pragma link C++ class std::vector < o2::itsmft::PhysTrigger> + ;

#pragma link C++ class o2::itsmft::CTFHeader + ;
#pragma link C++ class o2::itsmft::CompressedClusters + ;
#pragma link C++ class o2::itsmft::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::itsmft::CTFHeader, 10, uint32_t> + ;

#pragma link C++ class std::map < unsigned long, std::vector < uint16_t>> + ;

#pragma link C++ function o2::itsmft::getROFData(const gsl::span <const o2::itsmft::Digit> tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const gsl::span <const o2::itsmft::Cluster> tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const gsl::span <const o2::itsmft::CompCluster> tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const gsl::span <const o2::itsmft::CompClusterExt> tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const gsl::span < const o2::dataformats::Vertex <o2::dataformats::TimeStamp <int>> tfdata) const;

#pragma link C++ function o2::itsmft::getROFDataAt(int i, const gsl::span <const o2::itsmft::Digit> tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const gsl::span <const o2::itsmft::Cluster> tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const gsl::span <const o2::itsmft::CompCluster> tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const gsl::span <const o2::itsmft::CompClusterExt> tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const gsl::span <const o2::dataformats::Vertex <o2::dataformats::TimeStamp <int>>> tfdata) const;

#pragma link C++ function o2::itsmft::getROFData(const std::vector <o2::itsmft::Digit>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const std::vector <o2::itsmft::Cluster>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const std::vector <o2::itsmft::CompCluster>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const std::vector <o2::itsmft::CompClusterExt>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFData(const gsl::span < const o2::dataformats::Vertex <o2::dataformats::TimeStamp <int>>& tfdata) const;

#pragma link C++ function o2::itsmft::getROFDataAt(int i, const std::vector <o2::itsmft::Digit>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const std::vector <o2::itsmft::Cluster>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const std::vector <o2::itsmft::CompCluster>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const std::vector <o2::itsmft::CompClusterExt>& tfdata) const;
#pragma link C++ function o2::itsmft::getROFDataAt(int i, const std::vector <o2::dataformats::Vertex <o2::dataformats::TimeStamp <int>>>& tfdata) const;

#endif
