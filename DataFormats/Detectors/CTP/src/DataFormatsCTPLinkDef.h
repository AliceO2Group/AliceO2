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
#pragma link C++ class o2::ctp::CTPDigit + ;
#pragma link C++ class vector < o2::ctp::CTPDigit> + ;
#pragma link C++ class o2::ctp::CTPInputDigit + ;
#pragma link C++ class vector < o2::ctp::CTPInputDigit> + ;
#pragma link C++ class o2::ctp::BCMask + ;
#pragma link C++ class vector < o2::ctp::BCMask> + ;
#pragma link C++ class o2::ctp::CTPGenerator + ;
#pragma link C++ class vector < o2::ctp::CTPGenerator> + ;
#pragma link C++ class o2::ctp::CTPInput + ;
#pragma link C++ class vector < o2::ctp::CTPInput> + ;
#pragma link C++ class o2::ctp::CTPDescriptor + ;
#pragma link C++ class vector < o2::ctp::CTPDescriptor> + ;
#pragma link C++ class o2::ctp::CTPDetector + ;
#pragma link C++ class vector < o2::ctp::CTPDetector> + ;
#pragma link C++ class o2::ctp::CTPCluster + ;
#pragma link C++ class vector < o2::ctp::CTPCluster> + ;
#pragma link C++ class o2::ctp::CTPClass + ;
#pragma link C++ class vector < o2::ctp::CTPClass> + ;
#pragma link C++ class o2::ctp::CTPConfiguration + ;
#pragma link C++ class o2::ctp::CTPScalerRaw + ;
#pragma link C++ class vector < o2::ctp::CTPScalerRaw> + ;
#pragma link C++ class o2::ctp::CTPScalerO2 + ;
#pragma link C++ class vector < o2::ctp::CTPScalerO2> + ;
#pragma link C++ class o2::ctp::CTPScalerRecordRaw + ;
#pragma link C++ class vector < o2::ctp::CTPScalerRecordRaw> + ;
#pragma link C++ class o2::ctp::CTPScalerRecordO2 + ;
#pragma link C++ class vector < o2::ctp::CTPScalerRecordO2> + ;
#pragma link C++ class o2::ctp::CTPRunScalers + ;

#pragma link C++ struct o2::ctp::CTFHeader + ;
#pragma link C++ struct o2::ctp::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::ctp::CTFHeader, 4, uint32_t> + ;

#pragma link C++ class o2::ctp::TriggerOffsetsParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::ctp::TriggerOffsetsParam> + ;

#endif
