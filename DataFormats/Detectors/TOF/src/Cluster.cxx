// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.cxx
/// \brief Implementation of the TOF cluster

#include "DataFormatsTOF/Cluster.h"
#include "FairLogger.h"

#include <TMath.h>
#include <TString.h>

#include <cstdlib>

using namespace o2::tof;

ClassImp(o2::tof::Cluster);

Cluster::Cluster(std::int16_t sensid, float x, float y, float z, float sy2, float sz2, float syz, float timeRaw, float time, float tot, int L0L1Latency, int deltaBC) : o2::BaseCluster<float>(sensid, x, y, z, sy2, sz2, syz), mTimeRaw(timeRaw), mTime(time), mTot(tot), mL0L1Latency(L0L1Latency), mDeltaBC(deltaBC), mContributingChannels(0)
{

  // caching R and phi
  mR = TMath::Sqrt(x * x + y * y);
  mPhi = TMath::ATan2(y, x);
}

//______________________________________________________________________
int Cluster::getNumOfContributingChannels() const
{
  //
  // returning how many hits contribute to this cluster
  //
  int nContributingChannels = 0;
  if (mContributingChannels == 0) {
    LOG(ERROR) << "The current cluster has no hit contributing to it!" << FairLogger::endl;
  } else {
    nContributingChannels++;
    if ((mContributingChannels & kUpLeft) == kUpLeft)
      nContributingChannels++; //
    if ((mContributingChannels & kUp) == kUp)
      nContributingChannels++; // alsoDOWN
    if ((mContributingChannels & kUpRight) == kUpRight)
      nContributingChannels++; // alsoRIGHT
    if ((mContributingChannels & kRight) == kRight)
      nContributingChannels++; // alsoLEFT
    if ((mContributingChannels & kDownRight) == kDownRight)
      nContributingChannels++; // alsoL
    if ((mContributingChannels & kDown) == kDown)
      nContributingChannels++; // alsoLEFT
    if ((mContributingChannels & kDownLeft) == kDownLeft)
      nContributingChannels++; // alsoLEF
    if ((mContributingChannels & kLeft) == kLeft)
      nContributingChannels++; // alsoLEFT
  }
  return nContributingChannels;
}

//______________________________________________________________________
std::ostream& operator<<(std::ostream& os, const Cluster& c)
{
  os << (o2::BaseCluster<float>&)c;
  os << " TOF cluster: raw time = " << std::scientific << c.getTimeRaw() << ", time = " << std::scientific << c.getTime() << ", Tot = " << std::scientific << c.getTot() << ", L0L1Latency = " << c.getL0L1Latency() << ", deltaBC = " << c.getDeltaBC() << ", R = " << c.getR() << ", mPhi = " << c.getPhi() << ", ContributingChannels = " << c.getNumOfContributingChannels() << "\n";
  return os;
}
