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

#include <iostream>

using namespace o2::tof;

ClassImp(o2::tof::Digit);

Digit::Digit(Int_t channel, Int_t tdc, Int_t tot, Int_t bc, Int_t label)
  : mChannel(channel), mTDC(tdc), mTOT(tot), mBC(bc), mLabel(label), mIsUsedInCluster(kFALSE)
{
}

//______________________________________________________________________

void Digit::printStream(std::ostream& stream) const
{
  stream << "TOF Digit: Channel " << mChannel << " TDC " << mTDC << " TOT " << mTOT << "Bunch Crossing index" << mBC << " Label " << mLabel << "\n";
}

//______________________________________________________________________

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.printStream(stream);
  return stream;
}

//______________________________________________________________________

void Digit::merge(Int_t tdc, Int_t tot)
{

  // merging two digits

  if (tdc < mTDC) {
    mTDC = tdc;
    // TODO: adjust TOT
  } else {
    // TODO: adjust TOT
  }
}

//______________________________________________________________________

void Digit::getPhiAndEtaIndex(int& phi, int& eta)
{

  // method that returns the index in phi and eta of the digit

  int chan;
  int detId[5];
  chan = getChannel();                // note that inside the strip the digits are ordered per channel number
  Geo::getVolumeIndices(chan, detId); // Get volume index from channel index
  eta = detId[2] /*strip*/ * 2 + detId[3] /*pad Z*/;
  if (detId[1] /*module*/ == 0)
    eta += 0;
  else if (detId[1] == 1)
    eta += 38;
  else if (detId[1] == 2)
    eta += 76;
  else if (detId[1] == 3)
    eta += 106;
  else if (detId[1] == 4)
    eta += 144;
  phi = detId[0] /*phi sector*/ * 48 + detId[4] /*pad x*/;

  return;
}
