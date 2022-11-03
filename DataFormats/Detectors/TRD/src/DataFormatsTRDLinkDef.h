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

#pragma link C++ class o2::trd::TriggerRecord + ;
#pragma link C++ class o2::trd::TrackTriggerRecord + ;
#pragma link C++ class o2::trd::LinkRecord + ;
#pragma link C++ struct o2::trd::HalfCRUHeader + ;
#pragma link C++ struct o2::trd::TRDFeeID + ;
#pragma link C++ struct o2::trd::TrackletHCHeader + ;
#pragma link C++ struct o2::trd::TrackletMCMHeader + ;
#pragma link C++ struct o2::trd::TrackletMCMData + ;
// TODO add also digit data types and ClassDefNV etc...
#pragma link C++ class o2::trd::Tracklet64 + ;
#pragma link C++ class o2::trd::CalibratedTracklet + ;
#pragma link C++ class o2::trd::Hit + ;
#pragma link C++ class o2::trd::Digit + ;
#pragma link C++ class o2::trd::KrCluster + ;
#pragma link C++ class o2::trd::KrClusterTriggerRecord + ;
#pragma link C++ class o2::trd::NoiseStatusMCM + ;
#pragma link C++ class o2::trd::PadAdcInfo + ;
#pragma link C++ class o2::trd::AngularResidHistos + ;
#pragma link C++ class o2::trd::CalVdriftExB + ;
#pragma link C++ class o2::trd::CalT0 + ;
#pragma link C++ class o2::trd::CompressedDigit + ;
#pragma link C++ class o2::trd::HelperMethods + ;
#pragma link C++ class o2::trd::LinkToHCIDMapping + ;
#pragma link C++ class std::vector < o2::trd::Tracklet64> + ;
#pragma link C++ class std::vector < o2::trd::CalibratedTracklet> + ;
#pragma link C++ class std::vector < o2::trd::TrackTriggerRecord> + ;
#pragma link C++ class std::vector < o2::trd::TriggerRecord > +;
#pragma link C++ class std::vector < o2::trd::LinkRecord > +;
#pragma link C++ class std::vector < o2::trd::Hit > +;
#pragma link C++ class std::vector < o2::trd::Digit> + ;
#pragma link C++ class std::vector < o2::trd::AngularResidHistos> + ;
#pragma link C++ class std::vector < o2::trd::PadAdcInfo> + ;
#pragma link C++ class std::vector < o2::trd::KrCluster> + ;
#pragma link C++ class std::vector < o2::trd::KrClusterTriggerRecord> + ;

#pragma link C++ struct o2::trd::CTFHeader + ;
#pragma link C++ struct o2::trd::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::trd::CTFHeader, 15, uint32_t> + ;

#pragma link C++ struct o2::trd::TRDDCSMinMaxMeanInfo + ;

#endif
