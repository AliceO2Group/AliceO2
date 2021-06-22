// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalibWorkflow/TurnOnHistos.h"

using namespace o2::phos;

void TurnOnHistos::merge(TurnOnHistos& other)
{
  for (int i = NCHANNELS; --i;) {
    mGoodMap[i] += other.mGoodMap[i];
    mNoisyMap[i] += other.mNoisyMap[i];
  }
  for (int i = NDDL; --i;) {
    for (int j = Npt; --j;) {
      mTotSp[i][j] += other.mTotSp[i][j];
      mTrSp[i][j] += other.mTrSp[i][j];
    }
  }
}
