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
#include <gsl/span>
#include <numeric>
#include <string_view>
#include <utility>

//o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "DataFormatsTPC/Defs.h"
#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "MathUtils/Utils.h"
#include "TPCCalibration/FastHisto.h"

//root includes
#include "TFile.h"
#include "TH2F.h"

// boost includes
#include <boost/histogram.hpp>

using namespace o2::tpc;
namespace bh = boost::histogram;

CalibdEdx::CalibdEdx(int nBins, float minTotdEdx, float maxTotdEdx,
                     float minMaxdEdx, float maxMaxdEdx, const TrackCuts& cuts)
  : mCuts{cuts}, mNBins{nBins}
{
  mTotHist = bh::make_histogram(
    bh::axis::regular<>(nBins, minTotdEdx, maxTotdEdx, "dEdx"),
    HistIntAxis(0, SECTORSPERSIDE, "sector"),
    HistIntAxis(0, SIDES, "side"),
    HistIntAxis(0, GEMSTACKSPERSECTOR, "type"));

  mMaxHist = bh::make_histogram(
    bh::axis::regular<>(nBins, minMaxdEdx, maxMaxdEdx, "dEdx"),
    HistIntAxis(0, SECTORSPERSIDE, "sector"),
    HistIntAxis(0, SIDES, "side"),
    HistIntAxis(0, GEMSTACKSPERSECTOR, "type"));
}

void CalibdEdx::fill(const TrackTPC& track)
{
  // applying cuts
  if (track.hasBothSidesClusters() || (mApplyCuts && !mCuts.goodTrack(track))) {
    return;
  }

  const auto& dEdx = track.getdEdx();
  const auto side = track.hasASideClustersOnly() ? Side::A : Side::C;

  for (const GEMstack stack : {IROCgem, OROC1gem, OROC2gem, OROC3gem}) {
    bool ok = false;
    const int sector = findTrackSector(track, stack, ok);

    // Ignore stack if we are not able to find its sector
    if (!ok) {
      continue;
    }

    // Fill the correct readout type
    switch (stack) {
      case GEMstack::IROCgem:
        mTotHist(dEdx.dEdxTotIROC, sector, side, stack);
        mMaxHist(dEdx.dEdxMaxIROC, sector, side, stack);
        break;
      case GEMstack::OROC1gem:
        mTotHist(dEdx.dEdxTotOROC1, sector, side, stack);
        mMaxHist(dEdx.dEdxMaxOROC1, sector, side, stack);
        break;
      case GEMstack::OROC2gem:
        mTotHist(dEdx.dEdxTotOROC2, sector, side, stack);
        mMaxHist(dEdx.dEdxMaxOROC2, sector, side, stack);
        break;
      case GEMstack::OROC3gem:
        mTotHist(dEdx.dEdxTotOROC3, sector, side, stack);
        mMaxHist(dEdx.dEdxMaxOROC3, sector, side, stack);
        break;
    }
  }
}

void CalibdEdx::fill(const gsl::span<const TrackTPC> tracks)
{
  for (const auto& track : tracks) {
    fill(track);
  }
}

void CalibdEdx::fill(const std::vector<TrackTPC>& tracks)
{
  fill(gsl::span(tracks.data(), tracks.size()));
}

int CalibdEdx::findTrackSector(const TrackTPC& track, GEMstack stack, bool& ok)
{
  // These are the x value in cm of the center of the stacks (IROC, OROC1, ...) in the local frame.
  // FIXME: these values are only approximations.
  constexpr std::array<float, 4> xks{109.625f, 129.275f, 148.775f, 169.725f};
  constexpr float b = 0.5;

  float phi = track.getOuterParam().getXYZGloAt(xks[stack], b, ok).Phi();
  o2::math_utils::bringTo02PiGen(phi);

  return static_cast<int>(phi / SECPHIWIDTH);
}

void CalibdEdx::merge(const CalibdEdx* other)
{
  mTotHist += other->getFullHist(ChargeType::Tot);
  mMaxHist += other->getFullHist(ChargeType::Max);
}

void CalibdEdx::finalise()
{
  auto& calibEntries = mCalib.getEntries();
  size_t calibIndex = 0;
  std::vector<float> stackData(mNBins);

  for (const auto charge : {ChargeType::Tot, ChargeType::Max}) {
    const auto& hist = getFullHist(charge);

    const auto mindEdx = static_cast<float>(hist.axis(0).begin()->lower());
    const auto maxdEdx = static_cast<float>(hist.axis(0).end()->lower());
    int dEdxbin = 0;

    // std::vector<float> fitValues;
    for (const auto& bin : bh::indexed(hist)) {
      // Fit the Center of Gravity if we hit the end of a 1D dEdx histogram
      if (dEdxbin == mNBins) {
        // TODO: the gauss fit wasnt good. Noise from low count bins?
        // o2::math_utils::fitGaus(stackData.size(), stackData.data(), mindEdx, maxdEdx, fitValues);
        // calibEntries[calibIndex] = fitValues[1];

        const auto stats = o2::math_utils::getStatisticsData(stackData.data(), stackData.size(), mindEdx, maxdEdx);
        calibEntries[calibIndex] = stats.mCOG;
        ++calibIndex;
        dEdxbin = 0;
      }

      // FIXME: find a way to remove this copy to vector, maybe not possible.
      stackData[dEdxbin] = *bin;
      ++dEdxbin;
    }
  }
}

bool CalibdEdx::hasEnoughData(float minEntries) const
{
  using namespace bh::literals; // enables _c suffix

  // sum over the dEdx bins to find the number of entries per stack
  const auto projection = bh::algorithm::project(mTotHist, 1_c, 2_c, 3_c);
  auto dEdxCounts = bh::indexed(projection);
  return std::all_of(dEdxCounts.begin(), dEdxCounts.end(), [minEntries](const auto& x) { return x >= minEntries; });
}

TH2F CalibdEdx::getRootHist(ChargeType charge) const
{
  auto hist = getHist(charge);

  const float lower = hist.axis(0).begin()->lower();
  const float upper = hist.axis(0).end()->lower();

  const unsigned nbins = hist.axis(0).size();
  const unsigned nhists = hist.size() / nbins;

  TH2F rootHist("", "", nhists, 0, nhists, nbins, lower, upper);

  int stack = 0;
  float last_center = -1;
  // fill TH2
  for (auto&& x : bh::indexed(hist)) {
    const auto y = x.bin(0).center(); // current bin interval along dEdx axis
    const auto w = *x;                // "dereference" to get the bin value
    rootHist.Fill(stack, y, w);

    if (y < last_center) {
      stack++;
    }
    last_center = y;
  }

  return rootHist;
}


void CalibdEdx::print() const
{
  const int unique_entries = std::accumulate(mTotHist.begin(), mTotHist.end(), 0.0) / GEMSTACKSPERSECTOR / 2;
  LOGP(info, "Total number of track entries: {}", unique_entries);
}

void CalibdEdx::dumpToFile(std::string_view fileName) const
{
  TFile file(fileName.data(), "recreate");
  file.WriteObject(&mTotHist, "TotdEdxHist");
  file.WriteObject(&mMaxHist, "MaxdEdxHist");
  file.Close();
}
