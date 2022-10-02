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

/// \file Cluster.cxx
/// \brief Implementation of the TOF cluster

#include "DataFormatsTOF/Cluster.h"
#include <fairlogger/Logger.h>

#include <TString.h>

#include <cstdlib>

using namespace o2::tof;

ClassImp(o2::tof::Cluster);

Cluster::Cluster(std::int16_t sensid, float x, float y, float z, float sy2, float sz2, float syz, double timeRaw, double time, float tot, int L0L1Latency, int deltaBC) : o2::BaseCluster<float>(sensid, x, y, z, sy2, sz2, syz), mTimeRaw(timeRaw), mTime(time), mTot(tot), mL0L1Latency(L0L1Latency), mDeltaBC(deltaBC)
{

  // caching R and phi
  mR = o2::gpu::CAMath::Sqrt(x * x + y * y);
  mPhi = o2::gpu::CAMath::ATan2(y, x);
}
//______________________________________________________________________
int Cluster::getNumOfContributingChannels() const
{
  //
  // returning how many hits contribute to this cluster
  //
  int nContributingChannels = 1;

  if (isAdditionalChannelSet(kUpLeft)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kUp)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kUpRight)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kRight)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kDownRight)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kDown)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kDownLeft)) {
    nContributingChannels++;
  }
  if (isAdditionalChannelSet(kLeft)) {
    nContributingChannels++;
  }

  return nContributingChannels;
}

//______________________________________________________________________
std::ostream& operator<<(std::ostream& os, Cluster& c)
{
  os << (o2::BaseCluster<float>&)c;
  os << " TOF cluster: raw time = " << std::scientific << c.getTimeRaw() << ", time = " << std::scientific << c.getTime() << ", Tot = " << std::scientific << c.getTot() << ", L0L1Latency = " << c.getL0L1Latency() << ", deltaBC = " << c.getDeltaBC() << ", R = " << c.getR() << ", mPhi = " << c.getPhi() << ", Number of contributingChannels = " << c.getNumOfContributingChannels() << "\n";
  return os;
}

//______________________________________________________________________
void Cluster::setDigitInfo(int idig, int ch, double t, float tot)
{
  mDigitInfoCh[idig] = ch;
  mDigitInfoT[idig] = t;
  mDigitInfoTOT[idig] = tot;
}
