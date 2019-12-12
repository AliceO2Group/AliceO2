// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalib/TriggerDCS.h"
#include "EMCALCalib/TriggerTRUDCS.h"
#include "EMCALCalib/TriggerSTUDCS.h"

#include "FairLogger.h"

#include <bitset>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace o2::emcal;

bool TriggerDCS::operator==(const TriggerDCS& other) const
{
  return (mSTUEMCal == other.mSTUEMCal) && (mSTUDCAL == other.mSTUDCAL) && (mTRUArr == other.mTRUArr);
}

bool TriggerDCS::isTRUEnabled(int itru) const
{
  if (itru < 32) {
    return std::bitset<32>(mSTUEMCal.getRegion()).test(itru);
  } else {
    return std::bitset<32>(mSTUDCAL.getRegion()).test(itru - 32);
  }
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const TriggerDCS& config)
{
  stream << "EMCAL trigger DCS config:" << std::endl;
  stream << "================================" << std::endl;
  for (int i = 0; i < config.getTRUArr().size(); i++) {
    TriggerTRUDCS tru = config.getTRUDCS(i);
    stream << "TRU" << i << ": " << tru << std::endl;
  }
  TriggerSTUDCS emcalstu(config.getSTUDCSEMCal()), dcalstu(config.getSTUDCSDCal());
  std::cout << "EMCAL STU: " << emcalstu << std::endl;
  std::cout << "DCAL STU:  " << dcalstu << std::endl;
  return stream;
}

std::string TriggerDCS::toJSON() const
{
  std::stringstream jsonstring;
  jsonstring << "{";
  jsonstring << "\"mSTUEMCal\":" << mSTUEMCal.toJSON() << ",";
  jsonstring << "\"mSTUDCAL\":" << mSTUDCAL.toJSON() << ",";
  jsonstring << "mTRUArr:[";
  for (int ien = 0; ien < mTRUArr.size(); ien++) {
    jsonstring << "{\"TRU" << ien << "\":" << mTRUArr.at(ien).toJSON() << "}";
    if (ien != mTRUArr.size() - 1)
      jsonstring << ",";
  }
  jsonstring << "]}";
  return jsonstring.str();
}
