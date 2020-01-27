// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  Strip.cxx: structure to store the TOF digits in strips - useful
// for clusterization purposes
//  ALICEO2
//
#include <cstring>
#include <tuple>

#include <TMath.h>
#include <TObjArray.h>

#include "TOFBase/Strip.h"

using namespace o2::tof;

ClassImp(o2::tof::Strip);

int Strip::mDigitMerged = 0;

//_______________________________________________________________________
Strip::Strip(Int_t index)
  : mStripIndex(index)
{
}
//_______________________________________________________________________
Int_t Strip::addDigit(Int_t channel, Int_t tdc, Int_t tot, Int_t bc, Int_t lbl, Int_t triggerorbit, Int_t triggerbunch)
{

  // return the MC label. We pass it also as argument, but it can change in
  // case the digit was merged

  auto key = Digit::getOrderingKey(channel, bc, tdc); // the digits are ordered first per channel, then inside the channel per BC, then per time
  auto dig = findDigit(key);
  if (dig) {
    lbl = dig->getLabel(); // getting the label from the already existing digit
    dig->merge(tdc, tot);  // merging to the existing digit
    mDigitMerged++;
  } else {
    mDigits.emplace(std::make_pair(key, Digit(channel, tdc, tot, bc, lbl, triggerorbit, triggerbunch)));
  }

  return lbl;
}

//______________________________________________________________________
void Strip::fillOutputContainer(std::vector<Digit>& digits)
{
  // transfer digits that belong to the strip to the output array of digits
  // we assume that the Strip has stored inside only digits from one readout
  // window --> we flush them all

  if (mDigits.empty())
    return;
  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  for (; iter != mDigits.end(); ++iter) {
    Digit& dig = iter->second;
    digits.emplace_back(dig);
  }

  //  if (iter!=mDigits.end()) iter--;
  mDigits.erase(itBeg, iter);
}
