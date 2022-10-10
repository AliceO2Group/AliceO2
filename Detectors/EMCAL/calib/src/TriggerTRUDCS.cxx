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

#include "EMCALCalib/TriggerTRUDCS.h"

#include <fairlogger/Logger.h>

#include <bitset>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace o2::emcal;

TriggerTRUDCS::TriggerTRUDCS(uint64_t selpf, uint64_t l0sel, uint64_t l0cosm,
                             uint64_t gthrl0, uint64_t rlbkstu, uint64_t fw,
                             std::array<uint32_t, 6> maskReg) : mSELPF(selpf),
                                                                mL0SEL(l0sel),
                                                                mL0COSM(l0cosm),
                                                                mGTHRL0(gthrl0),
                                                                mRLBKSTU(rlbkstu),
                                                                mFw(fw),
                                                                mMaskReg(maskReg)
{
}

bool TriggerTRUDCS::operator==(const TriggerTRUDCS& other) const
{
  return (mSELPF == other.mSELPF) && (mL0SEL == other.mL0SEL) && (mL0COSM == other.mL0COSM) && (mGTHRL0 == other.mGTHRL0) && (mRLBKSTU == other.mRLBKSTU) && (mFw == other.mFw) && (mMaskReg == other.mMaskReg);
}

void TriggerTRUDCS::PrintStream(std::ostream& stream) const
{
  stream << "SELPF: 0x" << std::hex << mSELPF << ", L0SEL: 0x" << std::hex << mL0SEL << ", L0COSM: 0x" << std::hex
         << mL0COSM << ", GTHRL0: 0x" << std::hex << mGTHRL0 << ", RLBKSTU: 0x" << std::hex << mRLBKSTU << ", FW: 0x" << std::hex
         << mFw << std::dec << std::endl;

  for (int ireg = 0; ireg < 6; ireg++) {
    stream << "Reg" << ireg << ": b'" << std::bitset<sizeof(uint32_t) * 8>(mMaskReg[ireg]) << " (" << mMaskReg[ireg] << ")" << std::endl;
  }
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const TriggerTRUDCS& tru)
{
  tru.PrintStream(stream);
  return stream;
}

std::string TriggerTRUDCS::toJSON() const
{
  std::stringstream jsonstring;
  jsonstring << "{"
             << "\"mSELPF\":" << mSELPF << ","
             << "\"mL0SEL\":" << mL0SEL << ","
             << "\"mL0COSM\":" << mL0COSM << ","
             << "\"mGTHRL0\":" << mGTHRL0 << ","
             << "\"mRLBKSTU\":" << mRLBKSTU << ","
             << "\"mFw\":" << mFw << ","
             << "\"mMaskReg\":[" << mMaskReg[0] << "," << mMaskReg[1] << "," << mMaskReg[2] << "," << mMaskReg[3] << "," << mMaskReg[4] << "," << mMaskReg[5] << "]"
             << "}";
  return jsonstring.str();
}
