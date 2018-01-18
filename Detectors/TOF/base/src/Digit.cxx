// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFBase/Digit.h"

using namespace o2::tof;

ClassImp(o2::tof::Digit);

Digit::Digit(Double_t time, Int_t channel, Int_t tdc, Int_t tot, Int_t bc)
  : FairTimeStamp(time), mChannel(channel), mTDC(tdc), mTOT(tot), mBC(bc)
{
}

void Digit::printStream(std::ostream& stream) const
{
  stream << "TOF Digit: Channel " << mChannel << " TDC " << mTDC << " TOT " << mTOT << " Time " << GetTimeStamp()
         << "Bunch Crossing index" << mBC << "\n";
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.printStream(stream);
  return stream;
}
