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

#pragma link C++ class o2::fv0::Hit + ;
#pragma link C++ class vector < o2::fv0::Hit> + ;
#pragma link C++ class o2::fv0::MCLabel + ;

#pragma link C++ class o2::fv0::ChannelData + ;
#pragma link C++ class o2::fv0::BCData + ;
#pragma link C++ class o2::fv0::Triggers + ;
#pragma link C++ class o2::fv0::DetTrigInput + ;
#pragma link C++ class std::vector < o2::fv0::ChannelData> + ;
#pragma link C++ class std::vector < o2::fv0::Triggers> + ;
#pragma link C++ class std::vector < o2::fv0::DetTrigInput> + ;
#pragma link C++ class std::vector < o2::fv0::BCData> + ;

#pragma link C++ class o2::fv0::RawEventData + ;
#pragma link C++ class o2::fv0::EventHeader + ;
#pragma link C++ class o2::fv0::EventData + ;
#pragma link C++ class o2::fv0::TCMdata + ;
#pragma link C++ class o2::fv0::Topo + ;

#pragma link C++ class o2::fv0::CTFHeader + ;
#pragma link C++ class o2::fv0::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::fv0::CTFHeader, 6, uint32_t> + ;

#endif
