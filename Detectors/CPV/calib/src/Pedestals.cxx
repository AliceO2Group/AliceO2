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

using namespace o2::cpv;

Pedestals::Pedestals(int /*dummy*/)
{
  //produce reasonable objest for test purposes
  mPedestals.fill(40);
}

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
    if (h->GetBinContent(i) > 255) {
      LOG(ERROR) << "pedestal value too large:" << h->GetBinContent(i) << "can not be stored in char";
      continue;
    }
    mPedestals[i] = char(h->GetBinContent(i));
  }
  return true;
}
