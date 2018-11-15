// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHBase/Digit.h" //check if proper path
//check if anything else, like mapping needed
//to be added later: MC-truth and cluster-info
//maybe: bool whether after or before FEE simu?

#include <iostream>

using namespace o2::mch;

ClassImp(o2::mch::Digit);

Digit::Digit(uint32_t pad, Double_t adc, Double_t time)//check if long etc for pad needed
  : mPadID(pad), mADC(adc), mTime(time)
{
}

//______________________________________________________________________

void Digit::printStream(std::ostream& stream) const
{
  stream << "MCH Digit: PadID " << mPadID << " ADC " << mADC << " Time " << mTime << "\n";
  
}

//______________________________________________________________________

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.printStream(stream);
  return stream;
}

