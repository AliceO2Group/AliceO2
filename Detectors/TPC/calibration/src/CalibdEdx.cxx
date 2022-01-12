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
#include <boost/histogram/algorithm/project.hpp>
#include <cmath>
#include <cstddef>
#include <gsl/span>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "DataFormatsTPC/Defs.h"
#include "Framework/Logger.h"

// root includes
#include "TFile.h"
#include "TH2F.h"
#include "TTree.h"
#include "TLinearFitter.h"

// boost includes
#include <boost/histogram.hpp>
#include <vector>

using namespace o2::tpc;
namespace bh = boost::histogram;

CalibdEdx::CalibdEdx(float mindEdx, float maxdEdx, int dEdxBins, int zBins, int angularBins)
{
  constexpr float maxZ = 250;
  mHist = bh::make_histogram(
    FloatAxis(dEdxBins, mindEdx * mipScale, maxdEdx * mipScale, "dEdx"),
    FloatAxis(zBins, 0, maxZ, "Z"),
    FloatAxis(angularBins, -1, 1, "Tgl"),
    // HistFloatAxis(angleBins, -1, 1, "Snp"),
    IntAxis(0, SECTORSPERSIDE * SIDES, "sector"),
    IntAxis(0, GEMSTACKSPERSECTOR, "stackType"),
    IntAxis(0, CHARGETYPES, "charge"));
}

void CalibdEdx::fill(const TrackTPC& track)
{
  // applying cuts
  if (track.hasBothSidesClusters() || (mApplyCuts && !mCuts.goodTrack(track))) {
    return;
  }

  const auto& dEdx = track.getdEdx();
  const auto sideOffset = track.hasASideClustersOnly() ? 0 : SECTORSPERSIDE;
  const std::array<float, 4> dEdxMax{dEdx.dEdxMaxIROC, dEdx.dEdxMaxOROC1, dEdx.dEdxMaxOROC2, dEdx.dEdxMaxOROC3};
  const std::array<float, 4> dEdxTot{dEdx.dEdxTotIROC, dEdx.dEdxTotOROC1, dEdx.dEdxTotOROC2, dEdx.dEdxTotOROC3};
  // We need a copy of the track to perform propagations
  auto cpTrack = track;

  for (const GEMstack stack : {IROCgem, OROC1gem, OROC2gem, OROC3gem}) {
    // These are the x value in cm of the center of the stacks (IROC, OROC1, ...) in the local frame.
    constexpr std::array<float, 4> xks{108.475f, 151.7f, 188.8f, 227.65f};

    bool ok = cpTrack.propagateTo(xks[stack], mField);

    // Ignore stack if we are not able to find its sector
    if (!ok) {
      continue;
    }

    const float z = abs(cpTrack.getZ());
    const float tgl = cpTrack.getTgl();
    // const float snp = cpTrack.getSnp();
    const float alpha = o2::math_utils::to02PiGen(cpTrack.getAlpha());
    const auto sector = static_cast<int>(alpha / SECPHIWIDTH) + sideOffset;

    mHist(dEdxMax[stack] * mipScale, z, tgl, /* snp ,*/ sector, stack, ChargeType::Max);
    mHist(dEdxTot[stack] * mipScale, z, tgl, /* snp ,*/ sector, stack, ChargeType::Tot);
  }
}

void CalibdEdx::fill(const gsl::span<const TrackTPC> tracks)
{
  for (const auto& track : tracks) {
    fill(track);
  }
}

void CalibdEdx::merge(const CalibdEdx* other)
{
  if (other != nullptr) {
    mHist += other->getHist();
  }
}

template <typename Hist>
void fitHist(const Hist& hist, CalibdEdxCorrection& corr, TLinearFitter& fitter, const CalibdEdxCorrection* stackMean = nullptr)
{
  using ax = CalibdEdx::Axis;

  // number of bins per stack
  int stackBins = 1;
  for (int i = 0; i < ax::Sector; ++i) {
    stackBins *= hist.axis(i).size();
  }

  const bool projSectors = stackMean != nullptr;

  constexpr int sectors = SECTORSPERSIDE * SIDES;
  constexpr int stackCount = 144;
  // number of fits to perform
  const int fitCount = projSectors ? GEMSTACKSPERSECTOR * CHARGETYPES : stackCount * CHARGETYPES;
  // number of GEM stacks per fit
  const int fitStacks = projSectors ? sectors : 1;

  auto entry = bh::indexed(hist).begin();

  for (int fit = 0; fit < fitCount; ++fit) {
    StackID id{};
    id.type = static_cast<GEMstack>(entry->bin(ax::Stack).center());
    const auto charge = static_cast<ChargeType>(entry->bin(ax::Charge).center());

    for (int stack = 0; stack < fitStacks; ++stack) {
      id.sector = static_cast<int>(entry->bin(ax::Sector).center());

      for (int bin = 0; bin < stackBins; ++bin, ++entry) {
        const float counts = *entry;
        if (counts == 0) {
          continue;
        }
        std::array<double, 2> values{entry->bin(1).center(),
                                     entry->bin(2).center()};

        double dEdx = entry->bin(ax::dEdx).center();
        // scale fit using the stacks mean
        if (stackMean != nullptr) {
          dEdx /= stackMean->getCorrection(id, charge);
        }

        // constexpr float dEdxResolution = 0.05;
        // const double error = dEdx * dEdxResolution / sqrt(counts);
        const double error = 1. / sqrt(counts);

        fitter.AddPoint(values.data(), dEdx, error);
      }
    }
    fitter.Eval();

    constexpr auto paramSize = CalibdEdxCorrection::paramSize;
    float params[paramSize] = {0};
    for (int param = 0; param < fitter.GetNumberFreeParameters(); ++param) {
      params[param] = fitter.GetParameter(param);
    }

    // for projected hist, copy the fit to every sector
    if (projSectors) {
      for (int i = 0; i < sectors; ++i) {
        id.sector = i;
        const float mean = stackMean->getCorrection(id, charge);

        // rescale the params to get the true correction
        float scaledParams[paramSize];
        for (int i = 0; i < paramSize; ++i) {
          scaledParams[i] = params[i] * mean;
        }
        corr.setParams(id, charge, scaledParams);
        corr.setChi2(id, charge, fitter.GetChisquare());
      }
    } else {
      corr.setParams(id, charge, params);
      corr.setChi2(id, charge, fitter.GetChisquare());
    }
    fitter.ClearPoints();
  }
}

void CalibdEdx::finalize()
{
  const float entries = minStackEntries();
  mCalib.clear();

  TLinearFitter fitter(2);

  // Choose the fit dimension based on the available statistics
  if (entries >= mFitCuts[2]) {
    fitter.SetFormula("1 ++ x ++ x*x ++ y ++ x*y ++ y*y");
    mCalib.setDims(2);
  } else if (entries >= mFitCuts[1]) {
    fitter.SetFormula("1 ++ x ++ x*x");
    mCalib.setDims(1);
  } else {
    fitter.SetFormula("1");
    mCalib.setDims(0);
  }
  LOGP(info, "Fitting {}D dE/dx correction for GEM stacks", mCalib.getDims());

  // if entries bellow minimum threshold, integrate all sectors
  if (mCalib.getDims() == 0 || entries >= mFitCuts[0]) {
    fitHist(mHist, mCalib, fitter);
  } else {
    LOGP(info, "Integrating GEM stacks sectors in dE/dx correction due to low statistics");

    // get mean of each GEM stack
    CalibdEdxCorrection meanCorr{};
    meanCorr.setDims(0);
    TLinearFitter meanFitter(0);
    meanFitter.SetFormula("1");
    fitHist(mHist, meanCorr, meanFitter);

    // get highier dimension corrections with projected sectors
    fitHist(mHist, mCalib, fitter, &meanCorr);
  }
}

float CalibdEdx::minStackEntries() const
{
  // sum over the dEdx bins to get the number of entries per stack
  auto projection = bh::algorithm::project(mHist, std::vector<int>{Axis::Sector, Axis::Stack});
  auto dEdxCounts = bh::indexed(projection);
  // find the stack with the least number of entries
  auto min_it = std::min_element(dEdxCounts.begin(), dEdxCounts.end());
  // the count is doubled since we sum qMax and qTot entries
  return static_cast<float>(*min_it / 2);
}

bool CalibdEdx::hasEnoughData(float minEntries) const
{
  return minStackEntries() >= minEntries;
}

TH2F CalibdEdx::getRootHist(const std::vector<int>& projected_axis) const
{
  const float lower = mHist.axis(0).begin()->lower();
  const float upper = mHist.axis(0).end()->lower();

  auto projectedHist = getHist(projected_axis);

  const int nBins = mHist.axis(Axis::dEdx).size();
  const int nHists = projectedHist.size() / nBins;

  TH2F rootHist("hdEdxMIP", "MIP dEdx per GEM stack", nHists, 0, nHists, nBins, lower, upper);

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

TH2F CalibdEdx::getRootHist() const
{
  std::vector<int> keep_all(Axis::Size);
  std::iota(keep_all.begin(), keep_all.end(), 0);
  return getRootHist(keep_all);
}

void CalibdEdx::print() const
{
  const int uniqueEntries = std::accumulate(mHist.begin(), mHist.end(), 0.0) / GEMSTACKSPERSECTOR / 2;
  LOGP(info, "Total number of track entries: {}. Min. entries per GEM stack: {}", uniqueEntries, minStackEntries());
}

void CalibdEdx::writeTTree(std::string_view fileName) const
{
  TFile f(fileName.data(), "recreate");

  TTree tree("hist", "Saving boost histogram to TTree");

  std::vector<float> row(mHist.rank());
  for (int i = 0; i < mHist.rank(); ++i) {
    // FIXME: infer axis type and remove the hardcoded float
    tree.Branch(mHist.axis(i).metadata().c_str(), &row[i]);
  }
  float count = 0;
  tree.Branch("counts", &count);

  for (const auto& x : indexed(mHist)) {
    for (int i = 0; i < mHist.rank(); ++i) {
      row[i] = x.bin(i).center();
    }
    count = *x;
    tree.Fill();
  }

  f.Write();
  f.Close();
}
