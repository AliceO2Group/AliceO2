// Copyright CERN and copyright holders of ALICE O2.This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _FT0_CHANNELDATA_H_
#define _FT0_CHANNELDATA_H_

#include <Rtypes.h>

/// \file ChannelData.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author Alla.Maevskaya@cern.ch

namespace o2
{
namespace ft0
{
struct ChannelData {

  //public:

  int ChId = -1;     //channel Id
  int CFDTime = -1;  //time in #CFD channels, 0 at the LHC clk center
  int QTCAmpl = -1;  // Amplitude #channels
  int ChainQTC = -1; //QTC chain

  ChannelData() = default;
  ChannelData(int iPmt, int time, int charge, int chainQTC)
  {
    ChId = iPmt;
    CFDTime = time;
    QTCAmpl = charge;
    ChainQTC = chainQTC;
  }

  void print() const;
  
  ClassDefNV(ChannelData, 1);
};
} // namespace ft0
} // namespace o2
#endif
