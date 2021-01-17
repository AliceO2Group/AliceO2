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

#pragma link C++ class o2::ft0::Digit + ;
#pragma link C++ class o2::ft0::DigitExt + ;
#pragma link C++ class o2::ft0::DigitsTemp + ;
#pragma link C++ class o2::ft0::ChannelData + ;
#pragma link C++ class o2::ft0::Triggers + ;
#pragma link C++ class o2::ft0::DetTrigInput + ;
#pragma link C++ class o2::ft0::TriggersExt + ;
#pragma link C++ class vector < o2::ft0::ChannelData> + ;
#pragma link C++ class vector < o2::ft0::Digit> + ;
#pragma link C++ class vector < o2::ft0::DigitExt> + ;
#pragma link C++ class vector < o2::ft0::DigitsTemp> + ;
#pragma link C++ class vector < o2::ft0::Triggers> + ;
#pragma link C++ class vector < o2::ft0::DetTrigInput> + ;
#pragma link C++ class vector < o2::ft0::TriggersExt> + ;

#pragma link C++ class o2::ft0::RecPoints + ;
#pragma link C++ class vector < o2::ft0::RecPoints> + ;
#pragma link C++ class o2::ft0::ChannelDataFloat + ;
#pragma link C++ class vector < o2::ft0::ChannelDataFloat> + ;

#pragma link C++ class o2::ft0::MCLabel + ;
#include "SimulationDataFormat/MCTruthContainer.h"
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::ft0::MCLabel> + ;

#pragma link C++ class o2::ft0::HitType + ;
#pragma link C++ class vector < o2::ft0::HitType> + ;

#pragma link C++ class o2::ft0::RawEventData + ;
#pragma link C++ class o2::ft0::EventHeader + ;
#pragma link C++ class o2::ft0::EventData + ;
#pragma link C++ class o2::ft0::Topo + ;
#pragma link C++ class o2::ft0::HVchannel + ;
#pragma link C++ class vector < o2::ft0::HVchannel> + ;

#pragma link C++ class o2::ft0::CTFHeader + ;
#pragma link C++ class o2::ft0::CompressedDigits + ;
#pragma link C++ class o2::ft0::CTF + ;
#pragma link C++ class o2::ctf::EncodedBlocks < o2::ft0::CTFHeader, 9, uint32_t> + ;

#endif
