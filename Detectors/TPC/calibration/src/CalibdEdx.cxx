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
#include <vector>
#include <cmath>
#include <cstddef>
#include <gsl/span>
#include <numeric>
#include <string_view>
#include <utility>

// o2 includes
#include "CommonConstants/PhysicsConstants.h"
#include "DataFormatsTPC/BetheBlochAleph.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "Framework/Logger.h"
#include "TPCBase/ParameterGas.h"

// root includes
#include "TFile.h"
#include "THn.h"
#include "TTree.h"
#include "TLinearFitter.h"

// boost includes
#include <boost/histogram.hpp>

using namespace o2::tpc;
namespace bh = boost::histogram;

CalibdEdx::CalibdEdx(int dEdxBins, float mindEdx, float maxdEdx, int angularBins, bool fitSnp)
  : mFitSnp(fitSnp)
{
  const int snpBins = fitSnp ? angularBins : 1;
  mHist = bh::make_histogram(
    FloatAxis(dEdxBins, mindEdx * MipScale, maxdEdx * MipScale, "dEdx"),
    FloatAxis(angularBins, 0, 1, "Tgl"),
    FloatAxis(snpBins, -1, 1, "Snp"),
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
  o2::track::TrackPar cpTrack = track;

  // Beth-Bloch correction for non MIP tracks
  const auto& gasParam = ParameterGas::Instance();
  const float betaGamma = track.getP() / o2::constants::physics::MassPionCharged;
  const float dEdxScale = MipScale / BetheBlochAleph(betaGamma, gasParam.BetheBlochParam[0],
                                                     gasParam.BetheBlochParam[1], gasParam.BetheBlochParam[2],
                                                     gasParam.BetheBlochParam[3], gasParam.BetheBlochParam[4]);

  for (const GEMstack roc : {IROCgem, OROC1gem, OROC2gem, OROC3gem}) {
    // Local x value of the center pad row of each roc type in cm (IROC, OROC1, ...).
    constexpr std::array<float, 4> xks{108.475f, 151.7f, 188.8f, 227.65f};

    // propagate track
    const bool okProp = o2::base::Propagator::Instance()->PropagateToXBxByBz(cpTrack, xks[roc], 0.9f, 2., mMatType);
    if (!okProp) {
      continue;
    }

    // If the track was propagated to a different sector we need to rotate the local frame to get the correct Snp value
    float sector = std::floor(18.f * cpTrack.getPhiPos() / o2::constants::math::TwoPI);
    if (mFitSnp) {
      float localFrame = std::floor(18.f * o2::math_utils::to02PiGen(cpTrack.getAlpha()) / o2::constants::math::TwoPI);
      if (std::abs(sector - localFrame) > 0.1) {
        const float alpha = SECPHIWIDTH * (0.5 + sector);
        cpTrack.rotateParam(alpha);
      }
    }
    const float snp = cpTrack.getSnp();
    const float scaledTgl = scaleTgl(std::abs(cpTrack.getTgl()), roc);
    if (track.hasCSideClusters()) {
      sector += SECTORSPERSIDE;
    }

    mHist(dEdxMax[roc] * dEdxScale, scaledTgl, snp, sector, roc, ChargeType::Max);
    mHist(dEdxTot[roc] * dEdxScale, scaledTgl, snp, sector, roc, ChargeType::Tot);
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
void fitHist(const Hist& hist, CalibdEdxCorrection& corr, TLinearFitter& fitter,
             const float dEdxCut, const float dEdxLowCutFactor, const int passes, const CalibdEdxCorrection* stackMean = nullptr)
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

  for (int fitPass = 0; fitPass < passes; ++fitPass) {

    auto entry = bh::indexed(hist).begin();
    for (int fit = 0; fit < fitCount; ++fit) {
      int entries = 0;
      int outliers = 0;
      StackID id{};
      id.type = static_cast<GEMstack>(entry->bin(ax::Stack).center());
      const auto charge = static_cast<ChargeType>(entry->bin(ax::Charge).center());
      fitter.ClearPoints();

      for (int stack = 0; stack < fitStacks; ++stack) {
        id.sector = static_cast<int>(entry->bin(ax::Sector).center());

        for (int bin = 0; bin < stackBins; ++bin, ++entry) {
          const int counts = *entry;
          // skip empty bin
          if (counts == 0) {
            continue;
          }
          entries += counts;

          double dEdx = entry->bin(ax::dEdx).center();
          double inputs[] = {
            CalibdEdx::recoverTgl(entry->bin(ax::Tgl).center(), id.type),
            entry->bin(ax::Snp).center()};

          // ignore tracks with dEdx above a threshold defined by previous fit
          if (fitPass > 0) {
            float oldCorr = corr.getCorrection(id, charge, inputs[0], inputs[1]);
            float lowerCut = (1.f - dEdxLowCutFactor * dEdxCut) * oldCorr;
            float upperCut = (1.f + dEdxCut) * oldCorr;
            if (dEdx < lowerCut || dEdx > upperCut) {
              outliers += counts;
              continue;
            }
            // LOGP(info, "sector: {}, gemType: {}, bin: {}, fitPass: {}, oldCorr: {}, lowerCut: {}, upperCut: {}, dEdx: {}, counts: {}", id.sector, id.type, bin, fitPass, oldCorr, lowerCut, upperCut, dEdx, counts);
          }

          // scale fitted dEdx using the stacks mean
          if (stackMean != nullptr) {
            dEdx /= stackMean->getCorrection(id, charge);
          }
          const double error = 1. / sqrt(counts);
          fitter.AddPoint(inputs, dEdx, error);
        }
      }
      fitter.Eval();

      const auto paramSize = CalibdEdxCorrection::ParamSize;
      float params[paramSize] = {0};
      for (int param = 0; param < fitter.GetNumberFreeParameters(); ++param) {
        params[param] = fitter.GetParameter(param);
      }

      // with a projected hist, copy the fit to every sector
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
          corr.setEntries(id, charge, entries);
        }
      } else {
        corr.setParams(id, charge, params);
        corr.setChi2(id, charge, fitter.GetChisquare());
        corr.setEntries(id, charge, entries);
      }
      LOGP(debug, "Sector: {}, gemType: {}, charge: {}, Fit pass: {} with {} % outliers in {} entries. Fitter Points: {}, mean fit: {}",
           id.sector, int(id.type), int(charge), fitPass, (float)outliers / (float)entries * 100, entries, fitter.GetNpoints(), params[0]);
    }
  }
}

void CalibdEdx::finalize()
{
  const float entries = minStackEntries();
  mCalib.clear();

  TLinearFitter fitter(2);
  // Choose the fit dimension based on the available statistics
  if (mFitSnp && entries >= m2DThreshold) {
    fitter.SetFormula("1 ++ x ++ x*x ++ x*x*x ++ x*x*x*x ++ y ++ y*y ++ x*y");
    mCalib.setDims(2);
  } else if (entries >= m1DThreshold) {
    fitter.SetFormula("1 ++ x ++ x*x ++ x*x*x ++ x*x*x*x");
    mCalib.setDims(1);
  } else {
    fitter.SetFormula("1");
    mCalib.setDims(0);
  }
  LOGP(info, "Fitting {}D dE/dx correction for GEM stacks", mCalib.getDims());

  // if entries bellow minimum sector threshold, integrate all sectors
  if (mCalib.getDims() == 0 || entries >= mSectorThreshold) {
    fitHist(mHist, mCalib, fitter, mFitCut, mFitLowCutFactor, mFitPasses);
  } else {
    LOGP(info, "Integrating GEM stacks sectors in dE/dx correction due to low statistics");

    // get mean of each GEM stack
    CalibdEdxCorrection meanCorr{};
    meanCorr.setDims(0);
    TLinearFitter meanFitter(0);
    meanFitter.SetFormula("1");
    fitHist(mHist, meanCorr, meanFitter, mFitCut, mFitLowCutFactor, mFitPasses);

    // get higher dimension corrections with projected sectors
    fitHist(mHist, mCalib, fitter, mFitCut, mFitLowCutFactor, mFitPasses, &meanCorr);
  }
}

int CalibdEdx::minStackEntries() const
{
  // sum over the dEdx and track-param bins to get the number of entries per stack and charge
  auto projection = bh::algorithm::project(mHist, std::vector<int>{Axis::Sector, Axis::Stack, Axis::Charge});
  auto dEdxCounts = bh::indexed(projection);
  // find the stack with the least number of entries
  auto min_it = std::min_element(dEdxCounts.begin(), dEdxCounts.end());
  return *min_it;
}

bool CalibdEdx::hasEnoughData(float minEntries) const
{
  return minStackEntries() >= minEntries;
}

THnF* CalibdEdx::getRootHist() const
{
  std::vector<int> bins{};
  std::vector<double> axisMin{};
  std::vector<double> axisMax{};

  const size_t histRank = mHist.rank();

  for (size_t i = 0; i < histRank; ++i) {
    const auto& ax = mHist.axis(i);
    bins.push_back(ax.size());
    axisMin.push_back(*ax.begin());
    axisMax.push_back(*ax.end());
  }

  auto hn = new THnF("hdEdxMIP", "MIP dEdx per GEM stack", histRank, bins.data(), axisMin.data(), axisMax.data());
  std::vector<double> xs(histRank);
  for (auto&& entry : bh::indexed(mHist)) {
    if (*entry == 0) {
      continue;
    }
    for (int i = 0; i < histRank; ++i) {
      xs[i] = entry.bin(i).center();
    }

    hn->Fill(xs.data(), *entry);
  }
  return hn;
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

  // FIXME: infer axis type and remove the hardcoded float
  std::vector<float> row(mHist.rank());
  for (int i = 0; i < mHist.rank(); ++i) {
    tree.Branch(mHist.axis(i).metadata().c_str(), &row[i]);
  }
  float count = 0;
  tree.Branch("counts", &count);

  for (auto&& entry : bh::indexed(mHist)) {
    if (*entry == 0) {
      continue;
    }
    for (int i = 0; i < mHist.rank(); ++i) {
      // Rescale Tgl
      if (Axis::Tgl == i) {
        row[i] = recoverTgl(entry.bin(i).center(), static_cast<GEMstack>(entry.bin(Axis::Stack).center()));
      } else {
        row[i] = entry.bin(i).center();
      }
    }
    count = *entry;
    tree.Fill();
  }

  f.Write();
  f.Close();
}
