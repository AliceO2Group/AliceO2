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

#include "DataFormatsCPV/Pedestals.h"
#include <fairlogger/Logger.h>
#include <TH1.h>
#include <TH1F.h>

using namespace o2::cpv;

Pedestals::Pedestals(int /*dummy*/)
{
  //produce reasonable objest for test purposes
  mPedestals.fill(200); //typical pedestal value
  mPedSigmas.fill(1.5); //typical pedestal sigma
}
//______________________________________________________________________________
bool Pedestals::setPedestals(TH1* h)
{
  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS;
    return false;
  }

  for (short i = 1; i <= NCHANNELS; i++) {
    if (h->GetBinContent(i) > 511) {
      LOG(error) << "setPedestals : pedestal value = " << h->GetBinContent(i)
                 << " in channel " << i
                 << " exceeds max possible value 511 (limited by CPV electronics)";
      continue;
    }
    mPedestals[i - 1] = short(h->GetBinContent(i));
  }
  return true;
}
//_______________________________________________________________________________
bool Pedestals::setPedSigmas(TH1F* h)
{
  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS;
    return false;
  }

  for (short i = 1; i <= NCHANNELS; i++) {
    if (h->GetBinContent(i) < 0) {
      LOG(error) << "pedestal sigma = " << h->GetBinContent(i)
                 << " in channel " << i
                 << " cannot be less than 0";
      continue;
    }
    mPedSigmas[i - 1] = float(h->GetBinContent(i));
  }
  return true;
}
