// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/DigitsTemp.h"
#include <iostream>

using namespace o2::ft0;
void DigitsTemp::printStream(std::ostream& stream) const
{
  stream << "FT0 Digit: event time " << mTime << " BC " << mIntRecord.bc << " orbit " << mIntRecord.orbit << std::endl;
  stream << "IS A " << mTrigger.orA << " IS C " << mTrigger.orC << " is Central " << mTrigger.cen
         << " is SemiCentral " << mTrigger.sCen << " is Vertex " << mTrigger.vertex << std::endl;

  for (auto& chdata : mChDgDataArr)
    stream << "CH " << chdata.ChId << " TIME " << chdata.CFDTime << " ns " << chdata.QTCAmpl << " mV "
           << " ADC chain " << chdata.ChainQTC << std::endl;
}

std::ostream& operator<<(std::ostream& stream, const DigitsTemp& digi)
{
  digi.printStream(stream);
  return stream;
}
