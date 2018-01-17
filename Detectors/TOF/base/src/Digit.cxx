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
// ClassImp(Digit)

Digit::Digit(Int_t channel, Int_t tdc, Double_t time)
    : FairTimeStamp(time), mChannel(channel), mTDC(tdc) {}

void Digit::printStream(std::ostream &stream) const {
  stream << "TOF Digit: Channel " << mChannel << " TDC " << mTDC << ", Time "
         << GetTimeStamp();
}

std::ostream &operator<<(std::ostream &stream, const Digit &digi) {
  digi.printStream(stream);
  return stream;
}
