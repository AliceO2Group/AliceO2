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

/// \file Scalers.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Scalers.h"
#include <iostream>

using namespace o2::ctp;

void CTPScalerRaw::printStream(std::ostream& stream) const
{
  stream << "RAW LMB:" << lmBefore << " LMA:" << lmAfter;
  stream << " LOB:" << lmBefore << " L0A:" << lmAfter;
  stream << " L1B:" << lmBefore << " L1A:" << lmAfter << std::endl;
}
void CTPScalerO2::printStream(std::ostream& stream) const
{
  stream << "O2 LMB:" << lmBefore << " LMA:" << lmAfter;
  stream << " LOB:" << lmBefore << " L0A:" << lmAfter;
  stream << " L1B:" << lmBefore << " L1A:" << lmAfter << std::endl;
}
void CTPScalerRecordRaw::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc << std::endl;
  for(auto const& cnts: scalers) {
      cnts.printStream(stream);
  }
}
void CTPScalerRecordO2::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc << std::endl;
  for(auto const& cnts: scalers) {
      cnts.printStream(stream);
  }
}
void CTPRunScalers::printStream(std::ostream& stream) const
{
  stream << "CTP Scalers (version" << mVersion << ") Run:" << mRunNumber << std::endl;
  for(auto const& rec: mScalerRecordRaw) {
      rec.printStream(stream);
  }
}
void CTPRunScalers::printClasses(std::ostream& stream) const
{
  for(int i=0;i<mClassMask.size(); i++) {
  }
}
