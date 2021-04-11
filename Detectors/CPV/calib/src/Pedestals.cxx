// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVCalib/Pedestals.h"
#include "FairLogger.h"
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
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS;
    return false;
  }

  for (short i = 1; i <= NCHANNELS; i++) {
    if (h->GetBinContent(i) > 511) {
      LOG(ERROR) << "setPedestals : pedestal value = " << h->GetBinContent(i)
                 << " in channel " << i
                 << " exceeds max possible value 511 (limited by CPV electronics)";
      continue;
    }
    mPedestals[i] = short(h->GetBinContent(i));
  }
  return true;
}
//_______________________________________________________________________________
bool Pedestals::setPedSigmas(TH1F* h)
{
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != NCHANNELS) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << " instead of " << NCHANNELS;
    return false;
  }

  for (short i = 1; i <= NCHANNELS; i++) {
    if (h->GetBinContent(i) < 0) {
      LOG(ERROR) << "pedestal sigma = " << h->GetBinContent(i)
                 << " in channel " << i
                 << " cannot be less than 0";
      continue;
    }
    mPedSigmas[i] = float(h->GetBinContent(i));
  }
  return true;
}
