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

#include "TPCCalibration/CalibdEdx.h"

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
#include "DataFormatsTPC/Defs.h"
#include "Framework/Logger.h"
#include "TPCCalibration/FastHisto.h"

//root includes
#include "TFile.h"

using namespace o2::tpc;

CalibdEdx::CalibdEdx(unsigned int nBins, float mindEdx, float maxdEdx, const TrackCuts& cuts)
  : mCuts{cuts}, mNBins{nBins}
{
  const Hist emptyHist(nBins, mindEdx, maxdEdx, false, false);
  mHistos.init(emptyHist);
}

void CalibdEdx::fill(const gsl::span<const TrackTPC> tracks)
{
  auto& entries = mHistos.getEntries();

  for (const auto& track : tracks) {

    // applying cuts
    if (!track.hasBothSidesClusters() && (!mApplyCuts || mCuts.goodTrack(track))) {
      bool ok = false;
      const auto stacks = findTrackStacks(track, ok);

      // Ignore track if we are not able to find its stacks numbers
      if (ok) {
        const auto& trackdEdx = track.getdEdx();

        constexpr auto chargeOffset = HistContainer::totalStacks;

        // Fill each readout type
        // The Max data are stored after the Tot data, and so its indexes need to be incresead by chargeOffset
        // IROC
        mEntries[stacks[0]]++;
        entries[stacks[0]].fill(trackdEdx.dEdxTotIROC);
        entries[stacks[0] + chargeOffset].fill(trackdEdx.dEdxMaxIROC);

        // OROC1
        mEntries[stacks[1]]++;
        entries[stacks[1]].fill(trackdEdx.dEdxTotOROC1);
        entries[stacks[1] + chargeOffset].fill(trackdEdx.dEdxMaxOROC1);

        // OROC2
        mEntries[stacks[2]]++;
        entries[stacks[2]].fill(trackdEdx.dEdxTotOROC2);
        entries[stacks[2] + chargeOffset].fill(trackdEdx.dEdxMaxOROC2);

        // OROC3
        mEntries[stacks[3]]++;
        entries[stacks[3]].fill(trackdEdx.dEdxTotOROC3);
        entries[stacks[3] + chargeOffset].fill(trackdEdx.dEdxMaxOROC3);
      }
    }
  }
}

std::array<size_t, 4> CalibdEdx::findTrackStacks(const TrackTPC& track, bool& ok)
{
  // FIXME: these values are only approximations.
  // These are the x value in cm of the center of the stacks (IROC, OROC1, ...) in the local frame.
  constexpr std::array<float, 4> xks{109.625f, 129.275f, 148.775f, 169.725f};
  constexpr float b = 0.5;

  std::array<size_t, 4> stacks{};

  for (int stackType = 0; stackType < 4; ++stackType) {

    float phi = track.getOuterParam().getXYZGloAt(xks[stackType], b, ok).Phi();

    // return ok = false if we cant find any of the stacks numbers
    if (!ok) {
      break;
    }
    if (phi < 0) {
      phi = TWOPI + phi;
    }

    const auto sector = static_cast<size_t>(phi / SECPHIWIDTH);
    const Side side = track.hasASideClustersOnly() ? Side::A : Side::C;
    const auto type = static_cast<GEMstack>(stackType);

    // We get the Tot data because it is stored first in the container array.
    stacks[stackType] = HistContainer::stackIndex(sector, side, type, HistContainer::Charge::Tot);
  }

  return stacks;
}

void CalibdEdx::mergeHistos(Hist& fist, const Hist& second)
{
  const auto binCount = fist.getNBins();
  for (size_t bin = 0; bin < binCount; bin++) {
    const auto bin_content = second.getBinContent(bin);
    fist.fillBin(bin, bin_content);
  }
}

void CalibdEdx::merge(const CalibdEdx* other)
{
  auto& thisHistos = mHistos.getEntries();
  const auto& otherHistos = other->getHistos().getEntries();

  for (size_t i = 0; i < thisHistos.size(); i++) {
    mergeHistos(thisHistos[i], otherHistos[i]);
  }
}

void CalibdEdx::finalise()
{
  const auto& processHistos = [](const Hist& hist) { return hist.getStatisticsData().mCOG; };

  const auto& histos = mHistos.getEntries();
  auto& calib = mCalib.getEntries();
  std::transform(histos.begin(), histos.end(), calib.begin(), processHistos);
}

bool CalibdEdx::hasEnoughData(size_t minEntries) const
{
  for (const auto entries : mEntries) {
    if (entries < minEntries) {
      return false;
    }
  }
  return true;
}

void CalibdEdx::print() const
{
  const int unique_entries = std::accumulate(mEntries.begin(), mEntries.end(), 0) / GEMSTACKSPERSECTOR;
  LOGP(info, "Total number of track entries: {}", unique_entries);
}

void CalibdEdx::dumpToFile(std::string_view fileName) const
{
  TFile file(fileName.data(), "recreate");
  file.WriteObject(&mHistos.getEntries(), "StackdEdxHistos");

  file.Close();
}
