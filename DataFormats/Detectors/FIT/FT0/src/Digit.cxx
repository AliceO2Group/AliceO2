// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/Digit.h"
#include <iostream>

using namespace o2::ft0;

void Digit::printStream(std::ostream& stream) const
{
  stream << "FIT Digit: event time " << mTime << " BC " << mIntRecord.bc << " orbit " << mIntRecord.orbit << std::endl;
  stream << "IS A " << mIsA << " IS C " << mIsC << " is Central " << mIsCentral
         << " is SemiCentral " << mIsSemiCentral << " is Vertex " << mIsVertex << std::endl;

  for (auto& chdata : mChDgDataArr)
    stream << "CH " << chdata.ChId << " TIME " << chdata.CFDTime << " MIP " << chdata.QTCAmpl << std::endl;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.printStream(stream);
  return stream;
}
