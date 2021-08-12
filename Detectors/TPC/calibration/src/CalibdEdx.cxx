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

//root includes
#include "TFile.h"
#include "TH2F.h"

// boost includes
#include <boost/histogram.hpp>

using namespace o2::tpc;
namespace bh = boost::histogram;

CalibdEdx::CalibdEdx(int nBins, float minTotdEdx, float maxTotdEdx, const TrackCuts& cuts)
  : mCuts{cuts}, mNBins{nBins}
{
  mHist = bh::make_histogram(
    bh::axis::regular<>(nBins, minTotdEdx, maxTotdEdx, "dEdx"),
    HistIntAxis(0, SECTORSPERSIDE, "sector"),
    HistIntAxis(0, SIDES, "side"),
    HistIntAxis(0, GEMSTACKSPERSECTOR, "stack type"),
    HistIntAxis(0, DEDXCHARGETYPES, "charge"));
}

void CalibdEdx::fill(const TrackTPC& track)
{
  // applying cuts
  if (track.hasBothSidesClusters() || (mApplyCuts && !mCuts.goodTrack(track))) {
    return;
  }

  const auto& dEdx = track.getdEdx();
  const auto side = track.hasASideClustersOnly() ? Side::A : Side::C;
  const std::array<float, 4> dEdxTot{dEdx.dEdxTotIROC, dEdx.dEdxTotOROC1, dEdx.dEdxTotOROC2, dEdx.dEdxTotOROC3};
  const std::array<float, 4> dEdxMax{dEdx.dEdxMaxIROC, dEdx.dEdxMaxOROC1, dEdx.dEdxMaxOROC2, dEdx.dEdxMaxOROC3};

  for (const GEMstack stack : {IROCgem, OROC1gem, OROC2gem, OROC3gem}) {
    bool ok = false;
    const int sector = findTrackSector(track, stack, ok);

    // Ignore stack if we are not able to find its sector
    if (!ok) {
      continue;
    }

    mHist(dEdxTot[stack], sector, side, stack, dEdxCharge::Tot);
    mHist(dEdxTot[stack], sector, side, stack, dEdxCharge::Max);
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
  mHist += other->getFullHist();
}

void CalibdEdx::finalize()
{
  std::vector<float> stackData(mNBins);

  const auto mindEdx = static_cast<float>(mHist.axis(0).begin()->lower());
  const auto maxdEdx = static_cast<float>(mHist.axis(0).end()->lower());

  auto indexed = bh::indexed(mHist);
  auto entry = indexed.begin();
  while (entry != indexed.end()) {
    const int sector = entry->bin(HistAxis::Sector).lower();
    const auto side = static_cast<enum Side>(entry->bin(HistAxis::Side).lower());
    const auto type = static_cast<GEMstack>(entry->bin(HistAxis::Stack).lower());
    const auto charge = static_cast<dEdxCharge>(entry->bin(HistAxis::Charge).lower());

    // to use a fit function we fist copy the data to a vector of float
    // TODO: can we avoid this copy?
    for (size_t i = 0; i < stackData.size(); ++i, ++entry) {
      stackData[i] = *entry;
    }

    // TODO: the gauss fit wasnt good. Noise from low count bins?
    // std::vector<float> fitValues(4);
    // o2::math_utils::fitGaus(stackData.size(), stackData.data(), mindEdx, maxdEdx, fitValues);
    // mCalib.at(sector, side, type, charge) = fitValues[1];

    const auto stats = o2::math_utils::getStatisticsData(stackData.data(), stackData.size(), mindEdx, maxdEdx);
    mCalib.at(sector, side, type, charge) = stats.mCOG;
  }
}

bool CalibdEdx::hasEnoughData(float minEntries) const
{
  using namespace bh::literals; // enables _c suffix

  // sum over the dEdx bins to find the number of entries per stack
  const auto projection = bh::algorithm::project(mHist, 1_c, 2_c, 3_c);
  auto dEdxCounts = bh::indexed(projection);
  return std::all_of(dEdxCounts.begin(), dEdxCounts.end(), [minEntries](const auto& x) { return x >= minEntries; });
}

TH2F CalibdEdx::getRootHist() const
{
  const float lower = mHist.axis(0).begin()->lower();
  const float upper = mHist.axis(0).end()->lower();

  auto projectedHist = getHist();

  const int nHists = projectedHist.size() / mNBins;

  TH2F rootHist("", "", nHists, 0, nHists, mNBins, lower, upper);

  int stack = 0;
  float last_center = -1;
  // fill TH2
  for (auto&& x : bh::indexed(projectedHist)) {
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
  const int unique_entries = std::accumulate(mHist.begin(), mHist.end(), 0.0) / GEMSTACKSPERSECTOR / 2;
  LOGP(info, "Total number of track entries: {}", unique_entries);
}

void CalibdEdx::dumpToFile(std::string_view fileName) const
{
  TFile file(fileName.data(), "recreate");
  const auto rootHist = getRootHist();
  file.WriteObject(&rootHist, "CalibHists");
  file.WriteObject(&mCalib, "CalibData");
  file.Close();
}
