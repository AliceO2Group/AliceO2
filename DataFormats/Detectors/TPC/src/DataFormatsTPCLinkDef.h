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

#pragma link C++ class o2::tpc::ClusterHardware + ;
#pragma link C++ class o2::tpc::ClusterHardwareContainer + ;
#pragma link C++ class o2::tpc::ClusterNative + ;
#pragma link C++ class o2::tpc::ClusterHardwareContainer8kb + ;
#pragma link C++ class o2::tpc::ClusterHardwareContainerFixedSize < 8192> + ;
#pragma link C++ class o2::tpc::ClusterNativeContainer + ;
#pragma link C++ class o2::tpc::Digit + ;
#pragma link C++ class o2::tpc::ZeroSuppressedContainer8kb + ;
#pragma link C++ class std::vector < o2::tpc::ClusterNative> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterNativeContainer> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterHardware> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterHardwareContainerFixedSize < 8192>> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterHardwareContainer8kb> + ;
#pragma link C++ class std::vector < o2::tpc::Digit> + ;
#pragma link C++ class std::vector < o2::tpc::ZeroSuppressedContainer8kb> + ;
#pragma link C++ class o2::tpc::TrackTPC + ;
#pragma link C++ class o2::tpc::LaserTrack + ;
#pragma link C++ class o2::tpc::LaserTrackContainer + ;
#pragma link C++ class std::vector < o2::tpc::TrackTPC> + ;
#pragma link C++ class std::vector < o2::tpc::LaserTrack> + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper::Reader + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper::TreeWriter + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper::TreeWriter::BranchData + ;
#pragma link C++ class std::vector < o2::tpc::ClusterNativeHelper::TreeWriter::BranchData> + ;
#pragma link C++ class o2::tpc::dEdxInfo + ;
#pragma link C++ class o2::tpc::CompressedClustersCounters + ;
#pragma link C++ class o2::tpc::CompressedClustersPtrs_x < unsigned char*, unsigned short*, unsigned int*> + ;
#pragma link C++ class o2::tpc::CompressedClustersPtrs + ;
#pragma link C++ class o2::tpc::CompressedClusters + ;
#pragma link C++ class o2::tpc::CompressedClustersROOT - ;
#pragma link C++ class o2::tpc::CTF + ;
#pragma link C++ class o2::tpc::CTFHeader + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::tpc::CTFHeader, 23, uint32_t> + ;
#pragma link C++ enum o2::tpc::StatisticsType;
#pragma link C++ class o2::tpc::TrackCuts + ;
#pragma link C++ class o2::tpc::KrCluster + ;
#pragma link C++ class std::vector < o2::tpc::KrCluster> + ;
#pragma link C++ class o2::tpc::LtrCalibData + ;
#pragma link C++ class std::vector < o2::tpc::LtrCalibData> + ;
#pragma link C++ class o2::tpc::VDriftCorrFact + ;
#pragma link C++ class std::vector < o2::tpc::VDriftCorrFact> + ;
#pragma link C++ class o2::tpc::CalibdEdxCorrection + ;
#pragma link C++ class o2::tpc::dcs::DataPoint < float> + ;
#pragma link C++ class std::vector < o2::tpc::dcs::DataPoint < float>> + ;
#pragma link C++ class o2::tpc::dcs::DataPointVector < float> + ;
#pragma link C++ class o2::tpc::dcs::Temperature + ;
#pragma link C++ class o2::tpc::dcs::Temperature::Stats + ;
#pragma link C++ class o2::tpc::dcs::DataPoint < o2::tpc::dcs::Temperature::Stats> + ;
#pragma link C++ class o2::tpc::dcs::DataPointVector < o2::tpc::dcs::Temperature::Stats> + ;
#pragma link C++ class o2::tpc::dcs::HV + ;
#pragma link C++ class o2::tpc::dcs::DataPoint < o2::tpc::dcs::HV::StackState> + ;
#pragma link C++ class o2::tpc::dcs::DataPointVector < o2::tpc::dcs::HV::StackState> + ;
#pragma link C++ class o2::tpc::dcs::Gas + ;

#endif
