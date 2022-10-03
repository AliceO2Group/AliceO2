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

#include "DataFormatsPHOS/Pedestals.h"
#include <fairlogger/Logger.h>
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
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS + OFFSET) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS + OFFSET;
    return false;
  }

  for (short i = 0; i < NCHANNELS; i++) {
    if (h->GetBinContent(i + OFFSET) > 255) {
      LOG(error) << "pedestal value too large:" << h->GetBinContent(i + OFFSET) << "can not be stored in char";
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
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS + OFFSET) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS + OFFSET;
    return false;
  }

  for (short i = 0; i < NCHANNELS; i++) {
    if (h->GetBinContent(i + OFFSET) > 255) {
      LOG(error) << "pedestal value too large:" << h->GetBinContent(i + OFFSET) << "can not be stored in char";
      continue;
    }
    mLGPedestals[i] = static_cast<unsigned char>(h->GetBinContent(i + OFFSET));
  }
  return true;
}
