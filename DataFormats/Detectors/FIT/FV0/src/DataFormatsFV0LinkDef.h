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

#pragma link C++ class o2::fv0::Hit + ;
#pragma link C++ class vector < o2::fv0::Hit> + ;
#pragma link C++ class o2::fv0::MCLabel + ;

#pragma link C++ class o2::fv0::ChannelData + ;
#pragma link C++ class o2::fv0::Digit + ;
#pragma link C++ class o2::fv0::DetTrigInput + ;
#pragma link C++ class std::vector < o2::fv0::ChannelData> + ;
#pragma link C++ class std::vector < o2::fv0::DetTrigInput> + ;
#pragma link C++ class std::vector < o2::fv0::Digit> + ;

#pragma link C++ class o2::fv0::RawEventData + ;
#pragma link C++ class o2::fv0::CTFHeader + ;
#pragma link C++ class o2::fv0::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::fv0::CTFHeader, 8, uint32_t> + ;

#pragma link C++ class o2::fv0::RecPoints + ;
#pragma link C++ class vector < o2::fv0::RecPoints> + ;
#pragma link C++ class o2::fv0::ChannelDataFloat + ;
#pragma link C++ class vector < o2::fv0::ChannelDataFloat> + ;

#pragma link C++ class o2::fv0::FV0CalibrationInfoObject + ;
#pragma link C++ class o2::fv0::FV0ChannelTimeCalibrationObject + ;

#endif
