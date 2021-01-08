// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalib/TriggerTRUDCS.h"

#include "FairLogger.h"

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
  stream << "SELPF: " << std::hex << mSELPF << ", L0SEL: " << mL0SEL << ", L0COSM: " << std::dec
         << mL0COSM << ", GTHRL0: " << mGTHRL0 << ", RLBKSTU: " << mRLBKSTU << ", FW: " << std::hex
         << mFw << std::dec << std::endl;
  for (int ireg = 0; ireg < 6; ireg++) {
    stream << "Reg" << ireg << ": " << std::bitset<sizeof(uint32_t) * 8>(mMaskReg[ireg]) << " (" << mMaskReg[ireg] << ")" << std::endl;
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
