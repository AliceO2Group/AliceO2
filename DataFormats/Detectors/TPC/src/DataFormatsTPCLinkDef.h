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

#pragma link C++ class o2::tpc::ClusterHardware + ;
#pragma link C++ class o2::tpc::ClusterHardwareContainer + ;
#pragma link C++ class o2::tpc::ClusterNative + ;
#pragma link C++ class o2::tpc::ClusterHardwareContainer8kb + ;
#pragma link C++ class o2::tpc::ClusterHardwareContainerFixedSize < 8192> + ;
#pragma link C++ class o2::tpc::ClusterNativeContainer + ;
#pragma link C++ class std::vector < o2::tpc::ClusterNative> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterNativeContainer> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterHardware> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterHardwareContainerFixedSize < 8192>> + ;
#pragma link C++ class std::vector < o2::tpc::ClusterHardwareContainer8kb> + ;
#pragma link C++ class o2::tpc::TrackTPC + ;
#pragma link C++ class std::vector < o2::tpc::TrackTPC> + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper::Reader + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper::TreeWriter + ;
#pragma link C++ class o2::tpc::ClusterNativeHelper::TreeWriter::BranchData + ;
#pragma link C++ class std::vector < o2::tpc::ClusterNativeHelper::TreeWriter::BranchData> + ;
#pragma link C++ class o2::tpc::dEdxInfo + ;
#pragma link C++ class o2::tpc::CompressedClustersCounters + ;
#pragma link C++ class o2::tpc::CompressedClustersPtrs_helper < o2::tpc::CompressedClustersCounters> + ;
#pragma link C++ class o2::tpc::CompressedClusters + ;

#endif
