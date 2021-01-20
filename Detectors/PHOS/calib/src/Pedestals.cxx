// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalib/Pedestals.h"
#include "FairLogger.h"
#include <TH1.h>

using namespace o2::phos;

Pedestals::Pedestals(int /*dummy*/)
{
  //produce reasonable objest for test purposes
  mHGPedestals.fill(40);
  mLGPedestals.fill(35);
}

bool Pedestals::setHGPedestals(TH1* h)
{
  //We assume that histogram if filled vs absId of channels
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS + OFFSET) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS + OFFSET;
    return false;
  }

  for (short i = 0; i < NCHANNELS; i++) {
    if (h->GetBinContent(i + OFFSET) > 255) {
      LOG(ERROR) << "pedestal value too large:" << h->GetBinContent(i + OFFSET) << "can not be stored in char";
      continue;
    }
    mHGPedestals[i] = static_cast<unsigned char>(h->GetBinContent(i + OFFSET));
  }
  return true;
}
bool Pedestals::setLGPedestals(TH1* h)
{
  //We assume that histogram if filled vs absId of channels
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS + OFFSET) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS + OFFSET;
    return false;
  }

  for (short i = 0; i < NCHANNELS; i++) {
    if (h->GetBinContent(i + OFFSET) > 255) {
      LOG(ERROR) << "pedestal value too large:" << h->GetBinContent(i + OFFSET) << "can not be stored in char";
      continue;
    }
    mLGPedestals[i] = static_cast<unsigned char>(h->GetBinContent(i + OFFSET));
  }
  return true;
}
