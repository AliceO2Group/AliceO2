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

#include <sstream>

#include "EMCALCalib/FeeDCS.h"

using namespace o2::emcal;

bool FeeDCS::operator==(const FeeDCS& other) const
{

  return ((mTrigDCS == other.mTrigDCS) && (mLinks0 == other.mLinks0) && (mLinks1 == other.mLinks1) &&
          (mSRUFWversion == other.mSRUFWversion) && (mSRUcfg == other.mSRUcfg) && (mRunNumber == other.mRunNumber));
}

bool FeeDCS::isSMactive(int iSM)
{

  // assert(iSM>19 && "SM index larger than 19!");

  if (iSM == 10 || iSM == 19) { // SMs 10 and 19 have 1 DDL each
    return isDDLactive(2 * iSM);
  } else {
    return (isDDLactive(2 * iSM) && isDDLactive(2 * iSM + 1));
  }

  return false;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const FeeDCS& dcs)
{
  stream << "================================\n";
  stream << "EMCAL FEE config for run #" << std::dec << dcs.getRunNumber() << ": \n ";
  stream
    << "DDL Link list0: b'"
    << std::hex
    << dcs.getDDLlist0()
    << std::endl;
  stream << "DDL Link list1: b'" << std::hex << dcs.getDDLlist1() << std::endl;

  for (int i = 0; i < 20; i++) {
    stream << "SM" << std::dec << i << ": FW=0x" << std::hex << dcs.getSRUFWversion(i) << ", CFG=0x" << std::hex << dcs.getSRUconfig(i) << " [MEB=" << std::dec << dcs.getNSRUbuffers() << "] " << std::endl;
  }
  o2::emcal::TriggerDCS trg = dcs.getTriggerDCS();
  stream << trg << std::endl;

  return stream;
}
