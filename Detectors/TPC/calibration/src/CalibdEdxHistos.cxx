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

#include "TPCCalibration/CalibdEdxHistos.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string_view>
#include <utility>

//o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "Framework/Logger.h"
#include "TPCCalibration/FastHisto.h"

//root includes
#include "TFile.h"

using namespace o2::tpc;

CalibdEdxHistos::CalibdEdxHistos(unsigned int nBins, float mindEdx, float maxdEdx, const TrackCuts& cuts)
  : mCuts{cuts}, mNBins{nBins}
{
  const Hist emptyHist(nBins, mindEdx, maxdEdx, false, false);
  mTotEntries.fill(emptyHist);
  mMaxEntries.fill(emptyHist);
}

void CalibdEdxHistos::fill(const gsl::span<const TrackTPC> tracks)
{
  for (const auto& track : tracks) {

    // applying cuts
    if (!track.hasBothSidesClusters() && (!mApplyCuts || mCuts.goodTrack(track))) {
      bool ok = false;
      const auto stacks = findTrackStacks(track, ok);

      // Ignore track if we are not able to find its stacks numbers
      if (ok) {
        // Fill each readout type
        // IROC
        mEntries[stacks[0]]++;
        mTotEntries[stacks[0]].fill(track.getdEdx().dEdxTotIROC);
        mMaxEntries[stacks[0]].fill(track.getdEdx().dEdxMaxIROC);

        // OROC1
        mEntries[stacks[1]]++;
        mTotEntries[stacks[1]].fill(track.getdEdx().dEdxTotOROC1);
        mMaxEntries[stacks[1]].fill(track.getdEdx().dEdxMaxOROC1);

        // OROC2
        mEntries[stacks[2]]++;
        mTotEntries[stacks[2]].fill(track.getdEdx().dEdxTotOROC2);
        mMaxEntries[stacks[2]].fill(track.getdEdx().dEdxMaxOROC2);

        // OROC3
        mEntries[stacks[3]]++;
        mTotEntries[stacks[3]].fill(track.getdEdx().dEdxTotOROC3);
        mMaxEntries[stacks[3]].fill(track.getdEdx().dEdxMaxOROC3);
      }
    }
  }
}

std::array<float, 4> CalibdEdxHistos::findTrackStacks(const TrackTPC& track, bool& ok)
{
  // FIXME: these values are only approximations.
  // These are the x value in cm of the center of the stacks (IROC, OROC1, ...) in the local frame.
  constexpr std::array<float, 4> xks{109.625f, 129.275f, 148.775f, 169.725f};
  constexpr float b = 0.5;

  std::array<float, 4> stacks{};

  for (unsigned short typeNumber = 0; typeNumber < 4; ++typeNumber) {

    float phi = track.getOuterParam().getXYZGloAt(xks[typeNumber], b, ok).Phi();

    // return ok = false if we cant find any of the stacks numbers
    if (!ok) {
      break;
    }
    // LOG(INFO) << "Phi " << phi;
    constexpr float twoPi = 2 * M_PI;
    if (phi < 0) {
      phi = twoPi + phi;
    }

    constexpr float deltaPhi = twoPi / stacksPerTurn;
    const auto stack = static_cast<size_t>(phi / deltaPhi);
    // LOG(INFO) << "Stack number " << stack;

    const auto type = static_cast<ReadoutType>(typeNumber);
    const TPCSide side = track.hasASideClustersOnly() ? TPCSide::A : TPCSide::C;

    stacks[typeNumber] = stackIndex(stack, type, side);
  }

  return stacks;
}

void CalibdEdxHistos::mergeContainer(Container& fist, const Container& second)
{
  for (size_t i = 0; i < fist.size(); i++) {
    const auto binCount = fist[i].getNBins();
    for (size_t bin = 0; bin < binCount; bin++) {
      float bin_content = second[i].getBinContent(bin);
      fist[i].fillBin(bin, bin_content);
    }
  }
}

void CalibdEdxHistos::merge(const CalibdEdxHistos* other)
{
  mergeContainer(mTotEntries, other->getTotEntries());
  mergeContainer(mMaxEntries, other->getMaxEntries());
}

void CalibdEdxHistos::print() const
{
  const int unique_entries = std::accumulate(mEntries.begin(), mEntries.end(), 0) / 4;
  LOG(INFO) << "Total number of track entries: " << unique_entries;
}

void CalibdEdxHistos::dumpToFile(std::string_view fileName) const
{
  TFile file(fileName.data(), "recreate");
  file.WriteObject(&mTotEntries, "dEdxTotHistos");
  file.WriteObject(&mMaxEntries, "dEdxMaxHistos");

  file.Close();
}
