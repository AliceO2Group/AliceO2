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
#include <string>
#include <vector>
#include <cmath>
#include <cstddef>
#include <gsl/span>
#include <numeric>
#include <string_view>
#include <utility>
#include <chrono>

// o2 includes
#include "CommonConstants/PhysicsConstants.h"
#include "DataFormatsTPC/BetheBlochAleph.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "Framework/Logger.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Utils.h"
#include "CommonUtils/BoostHistogramUtils.h"

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

CalibdEdx::CalibdEdx(const CalibdEdx& other)
{
  mFitSnp = other.mFitSnp;
  mApplyCuts = other.mApplyCuts;
  mCuts = other.mCuts;
  mSectorThreshold = other.mSectorThreshold;
  m1DThreshold = other.m1DThreshold;
  m2DThreshold = other.m2DThreshold;
  mFitCut = other.mFitCut;
  mFitLowCutFactor = other.mFitLowCutFactor;
  mFitPasses = other.mFitPasses;
  mTFID = other.mTFID;

  mHist = other.mHist;
  mCalib = other.mCalib;
  mCalibIn = other.mCalibIn;

  mMatType = other.mMatType;

  // debug streamer not copied on purpose
}

void CalibdEdx::fill(const TrackTPC& track)
{
  // applying cuts
  if (track.hasBothSidesClusters() || (mApplyCuts && !mCuts.goodTrack(track))) {
    return;
  }

  const auto& dEdx = track.getdEdx();
  const auto sideOffset = track.hasASideClustersOnly() ? 0 : SECTORSPERSIDE;
  const std::vector<float> dEdxMax{dEdx.dEdxMaxIROC, dEdx.dEdxMaxOROC1, dEdx.dEdxMaxOROC2, dEdx.dEdxMaxOROC3};
  const std::vector<float> dEdxTot{dEdx.dEdxTotIROC, dEdx.dEdxTotOROC1, dEdx.dEdxTotOROC2, dEdx.dEdxTotOROC3};
  std::vector<float> dEdxMaxCorr(4); // for deugging
  std::vector<float> dEdxTotCorr(4); // for deugging

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
    const float tgl = cpTrack.getTgl();
    const float scaledTgl = scaleTgl(std::abs(tgl), roc);
    if (track.hasCSideClusters()) {
      sector += SECTORSPERSIDE;
    }

    // undo previously done corrections, to allow for residual corrections
    // output will still be the full correction
    const float corrMax = mCalibIn.getCorrection(StackID{static_cast<int>(sector), roc}, ChargeType::Max, tgl, snp);
    const float corrTot = mCalibIn.getCorrection(StackID{static_cast<int>(sector), roc}, ChargeType::Tot, tgl, snp);
    dEdxMaxCorr[roc] = corrMax;
    dEdxTotCorr[roc] = corrTot;

    static bool reported = false;
    if (!reported && mCalibIn.getDims() >= 0) {
      const auto meanParamTot = mCalibIn.getMeanParams(ChargeType::Tot);
      LOGP(info, "Undoing previously applied corrections with mean qTot Params {}", utils::elementsToString(meanParamTot));
      reported = true;
    }

    mHist(dEdxMax[roc] * dEdxScale * corrMax, scaledTgl, snp, sector, roc, ChargeType::Max);
    mHist(dEdxTot[roc] * dEdxScale * corrTot, scaledTgl, snp, sector, roc, ChargeType::Tot);
  }

  if (mDebugOutputStreamer) {
    const float tgl = track.getTgl();
    const float p = track.getP();

    (*mDebugOutputStreamer) << "dedx"
                            << "dEdxMax=" << dEdxMax
                            << "dEdxMaxCorr=" << dEdxMaxCorr
                            << "dEdxTot=" << dEdxTot
                            << "dEdxTotCorr=" << dEdxTotCorr
                            << "tgl=" << tgl
                            << "p=" << p
                            << "tfid=" << mTFID
                            << "\n";
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
             const float dEdxCut, const float dEdxLowCutFactor, const int passes, const CalibdEdxCorrection* stackMean = nullptr, o2::utils::TreeStreamRedirector* streamer = nullptr)
{
  using timer = std::chrono::high_resolution_clock;
  using ax = CalibdEdx::Axis;
  auto start = timer::now();

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

          if (streamer) {
            float oldCorr = corr.getCorrection(id, charge, inputs[0], inputs[1]);
            float lowerCut = (1.f - dEdxLowCutFactor * dEdxCut) * oldCorr;
            float upperCut = (1.f + dEdxCut) * oldCorr;

            (*streamer) << "fit_standard"
                        << "dedx=" << dEdx
                        << "itgl=" << hist.axis(ax::Tgl).index(entry->bin(ax::Tgl).center())
                        << "snp=" << inputs[1]
                        << "iter=" << fitPass
                        << "ifit=" << fit
                        << "bin=" << bin
                        << "isector=" << int(id.sector)
                        << "istack=" << int(id.type)
                        << "icharge=" << int(charge)
                        << "counts=" << counts
                        << "oldCorr=" << oldCorr
                        << "lowerCut=" << lowerCut
                        << "upperCut=" << upperCut
                        << "\n";
          }
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
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  LOGP(info, "Calibration fits took: {}", time.count());
}

template <typename Hist>
auto ProjectBoostHistoXFast(const Hist& hist, std::vector<int>& bin_indices, int axis)
{
  const unsigned int nbins = hist.axis(axis).size();
  const double binStartX = hist.axis(axis).bin(0).lower();
  const double binEndX = hist.axis(axis).bin(nbins - 1).upper();
  auto histoProj = boost::histogram::make_histogram(CalibdEdx::FloatAxis(nbins, binStartX, binEndX));

  // loop over all bins of the requested axis
  for (int i = 0; i < nbins; ++i) {
    // setting bin of the requested axis
    bin_indices[axis] = i;

    // access the bin content specified by bin_indices
    histoProj.at(i) = hist.at(bin_indices);
  }

  return histoProj;
}

template <typename Hist>
auto ProjectBoostHistoXFastAllSectors(const Hist& hist, std::vector<int>& bin_indices, StackID id, const CalibdEdxCorrection* stackMean)
{
  // get an average histogram over all stacks of the same type
  using ax = CalibdEdx::Axis;
  const unsigned int nbinsdedx = hist.axis(ax::dEdx).size();
  const double binStartX = hist.axis(ax::dEdx).bin(0).lower();
  const double binEndX = hist.axis(ax::dEdx).bin(nbinsdedx - 1).upper();

  // make fine histogram to be able to correctly store normalized dEdx values
  auto histoProj = boost::histogram::make_histogram(CalibdEdx::FloatAxis(nbinsdedx, binStartX, binEndX));

  // loop over sectors for merging the histograms
  for (int iSec = 0; iSec < hist.axis(ax::Sector).size(); ++iSec) {
    bin_indices[ax::Sector] = iSec;
    id.sector = static_cast<int>(hist.axis(ax::Sector).bin(iSec).center());

    // loop over all bins of the requested axis
    for (int i = 0; i < nbinsdedx; ++i) {
      // setting bin of the requested axis
      bin_indices[ax::dEdx] = i;

      // access the bin content specified by bin_indices
      const float counts = hist.at(bin_indices);
      float dEdx = hist.axis(ax::dEdx).bin(i).center();

      // scale the dedx to the mean
      if (stackMean != nullptr) {
        const auto charge = static_cast<ChargeType>(bin_indices[ax::Charge]);
        dEdx /= stackMean->getCorrection(id, charge);
      }

      // fill the histogram with dedx as the bin center and the counts as the weight
      histoProj(dEdx, bh::weight(counts));
    }
  }
  return histoProj;
}

void CalibdEdx::fitHistGaus(TLinearFitter& fitter, CalibdEdxCorrection& corr, const CalibdEdxCorrection* stackMean)
{
  using timer = std::chrono::high_resolution_clock;
  using ax = CalibdEdx::Axis;
  auto start = timer::now();
  const bool projSectors = stackMean != nullptr;
  constexpr int sectors = SECTORSPERSIDE * SIDES;
  for (int iSnp = 0; iSnp < mHist.axis(ax::Snp).size(); ++iSnp) {
    const int iSecEnd = projSectors ? 1 : mHist.axis(ax::Sector).size();
    for (int iSec = 0; iSec < iSecEnd; ++iSec) {
      for (int iStack = 0; iStack < mHist.axis(ax::Stack).size(); ++iStack) {
        for (int iCharge = 0; iCharge < mHist.axis(ax::Charge).size(); ++iCharge) {

          fitter.ClearPoints();
          StackID id{};
          id.type = static_cast<GEMstack>(mHist.axis(ax::Stack).bin(iStack).center());
          id.sector = static_cast<int>(mHist.axis(ax::Sector).bin(iSec).center());
          const auto charge = static_cast<ChargeType>(mHist.axis(ax::Charge).bin(iCharge).center());
          int entries = 0;

          for (int iTgl = 0; iTgl < mHist.axis(ax::Tgl).size(); ++iTgl) {
            // calculate sigma vs tgl in first iteration
            // apply cut in n sigma in second iteration
            float sigma_vs_tgl = 0;
            float mean_vs_tgl = 0;
            std::vector<int> bin_indices(ax::Size);
            bin_indices[ax::Tgl] = iTgl;
            bin_indices[ax::Snp] = iSnp;
            bin_indices[ax::Sector] = iSec;
            bin_indices[ax::Stack] = iStack;
            bin_indices[ax::Charge] = iCharge;

            for (int iter = 0; iter < 2; ++iter) {
              auto boostHist1d = projSectors ? ProjectBoostHistoXFastAllSectors(mHist, bin_indices, id, stackMean) : ProjectBoostHistoXFast(mHist, bin_indices, ax::dEdx);

              float lowerCut = 0;
              float upperCut = 0;

              // make gaussian fit
              if (iter == 0) {
                int maxElementIndex = std::max_element(boostHist1d.begin(), boostHist1d.end()) - boostHist1d.begin() - 1;
                if (maxElementIndex < 0) {
                  maxElementIndex = 0;
                }
                float maxElementCenter = 0.5 * (boostHist1d.axis(0).bin(maxElementIndex).upper() + boostHist1d.axis(0).bin(maxElementIndex).lower());

                lowerCut = (1.f - mFitLowCutFactor * mFitCut) * maxElementCenter;
                upperCut = (1.f + mFitCut) * maxElementCenter;
              } else {
                lowerCut = mean_vs_tgl - sigma_vs_tgl * mSigmaLower;
                upperCut = mean_vs_tgl + sigma_vs_tgl * mSigmaUpper;
              }

              // Restrict fit range to maximum +- restrictFitRangeToMax
              double max_range = mHist.axis(ax::dEdx).bin(mHist.axis(ax::dEdx).size() - 1).lower();
              double min_range = mHist.axis(ax::dEdx).bin(0).lower();
              if ((upperCut <= lowerCut) || (lowerCut > max_range) || (upperCut < min_range)) {
                break;
              }

              // remove up and low bins
              boostHist1d = boost::histogram::algorithm::reduce(boostHist1d, boost::histogram::algorithm::shrink(lowerCut, upperCut));

              try {
                auto fitValues = o2::utils::fitGaus<float>(boostHist1d.begin(), boostHist1d.end(), o2::utils::BinCenterView(boostHist1d.axis(0).begin()), false);

                // add the mean from gaus fit to the fitter
                double dEdx = fitValues[1];
                double inputs[] = {
                  CalibdEdx::recoverTgl(mHist.axis(ax::Tgl).bin(iTgl).center(), id.type),
                  mHist.axis(ax::Snp).bin(iSnp).center()};

                const auto fitNPoints = fitValues[3];
                const float sigma = fitValues[2];
                const double fitMeanErr = (fitNPoints > 0) ? (sigma / std::sqrt(fitNPoints)) : -1;
                if (iter == 0) {
                  sigma_vs_tgl = sigma;
                  mean_vs_tgl = dEdx;
                } else {
                  entries += fitNPoints;
                  if (fitMeanErr > 0) {
                    fitter.AddPoint(inputs, dEdx, fitMeanErr);
                  }
                }

                if (mDebugOutputStreamer) {
                  const int nbinsx = boostHist1d.axis(0).size();
                  std::vector<float> binCenter(nbinsx);
                  std::vector<float> dedx(nbinsx);
                  for (int ix = 0; ix < nbinsx; ix++) {
                    binCenter[ix] = boostHist1d.axis(0).bin(ix).center();
                    dedx[ix] = boostHist1d.at(ix);
                  }

                  (*mDebugOutputStreamer) << "fit_gaus"
                                          << "fitConstant=" << fitValues[0]
                                          << "fitMean=" << dEdx
                                          << "fitMeanErr=" << fitMeanErr
                                          << "fitSigma=" << sigma_vs_tgl
                                          << "fitSum=" << fitNPoints
                                          << "dedx=" << binCenter
                                          << "counts=" << dedx
                                          << "itgl=" << bin_indices[1]
                                          << "isnp=" << bin_indices[2]
                                          << "isector=" << bin_indices[3]
                                          << "istack=" << bin_indices[4]
                                          << "icharge=" << bin_indices[5]
                                          << "upperCut=" << upperCut
                                          << "lowerCut=" << lowerCut
                                          << "mFitCut=" << mFitCut
                                          << "mFitLowCutFactor=" << mFitLowCutFactor
                                          << "iter=" << iter
                                          << "mSigmaLower=" << mSigmaLower
                                          << "mSigmaUpper=" << mSigmaUpper
                                          << "projSectors=" << projSectors
                                          << "\n";
                }
              } catch (o2::utils::FitGausError_t err) {
                LOGP(warning, "Skipping bin: iTgl {} iSnp {} iSec {} iStack {} iCharge {} iter {}", iTgl, iSnp, iSec, iStack, iCharge, iter);
                LOG(warning) << createErrorMessageFitGaus(err);
                break;
              }
            }
          }

          const int fitStatus = fitter.Eval();
          if (fitStatus > 0) {
            LOGP(warning, "Fit failed");
            continue;
          }

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
        }
      }
    }
  }
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  LOGP(info, "Calibration fits took: {}", time.count());
}

void CalibdEdx::finalize(const bool useGausFits, const bool averageSectors)
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
  LOGP(info, "Fitting {}D dE/dx correction for GEM stacks with gaussian fits {}, minStackEntries {}, m2DThreshold {}, m1DThreshold {}, mFitSnp {}",
       mCalib.getDims(), useGausFits, entries, m2DThreshold, m1DThreshold, mFitSnp);

  // if entries below minimum sector threshold, integrate all sectors
  if (mCalib.getDims() == 0 || entries >= mSectorThreshold) {
    if (!useGausFits) {
      fitHist(mHist, mCalib, fitter, mFitCut, mFitLowCutFactor, mFitPasses, nullptr, mDebugOutputStreamer.get());
    } else {
      fitHistGaus(fitter, mCalib, nullptr);
    }
  } else {
    LOGP(info, "Integrating GEM stacks sectors in dE/dx correction due to low statistics");

    // get mean of each GEM stack
    CalibdEdxCorrection meanCorr{};
    meanCorr.setDims(0);
    if (averageSectors) {
      // set mean dEdx per stack to unity
      meanCorr.setUnity();
    } else {
      // get the mean dEdx for each stack
      TLinearFitter meanFitter(0);
      meanFitter.SetFormula("1");
      fitHist(mHist, meanCorr, meanFitter, mFitCut, mFitLowCutFactor, mFitPasses, nullptr, mDebugOutputStreamer.get());
    }
    if (!useGausFits) {
      // get higher dimension corrections with projected sectors
      fitHist(mHist, mCalib, fitter, mFitCut, mFitLowCutFactor, mFitPasses, &meanCorr, mDebugOutputStreamer.get());
    } else {
      // get higher dimension corrections with projected sectors
      fitHistGaus(fitter, mCalib, &meanCorr);
    }
  }
}

int CalibdEdx::minStackEntries() const
{
  // sum over the dEdx and track-param bins to get the number of entries per stack and charge
  auto projection = bh::algorithm::project(mHist, std::vector<int>{Axis::Sector, Axis::Stack, Axis::Charge});
  auto dEdxCounts = bh::indexed(projection);
  // find the stack with the least number of entries
  // use explicit int comparator to avoid ambiguous operator< between accessor and unlimited_storage::reference (boost/clang issue)
  auto min_it = std::min_element(dEdxCounts.begin(), dEdxCounts.end(),
                                 [](const auto& a, const auto& b) { return static_cast<int>(*a) < static_cast<int>(*b); });
  return *min_it;
}

bool CalibdEdx::hasEnoughData(float minEntries) const
{
  return minStackEntries() >= minEntries;
}

THnF* CalibdEdx::getTHnF() const
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
  return hn;
}

THnF* CalibdEdx::getRootHist() const
{
  auto hn = getTHnF();
  const size_t histRank = mHist.rank();
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

void CalibdEdx::setFromRootHist(const THnF* hist)
{
  // Get the number of dimensions
  int n_dim = hist->GetNdimensions();

  // Vectors to store axis ranges and bin counts
  std::vector<std::pair<double, double>> axis_ranges(n_dim); // Min and max of each axis
  std::vector<int> bin_counts(n_dim);                        // Number of bins in each dimension

  // Loop over each dimension to extract the bin edges and ranges
  for (int dim = 0; dim < n_dim; ++dim) {
    TAxis* axis = hist->GetAxis(dim);
    int bins = axis->GetNbins();
    double min = axis->GetXmin();
    double max = axis->GetXmax();
    bin_counts[dim] = bins;
    axis_ranges[dim] = {min, max}; // Store the min and max range for the axis
  }

  // Define a Boost histogram using the bin edges
  mHist = bh::make_histogram(
    FloatAxis(bin_counts[0], axis_ranges[0].first, axis_ranges[0].second, "dEdx"), // dE/dx
    FloatAxis(bin_counts[1], axis_ranges[1].first, axis_ranges[1].second, "Tgl"),  // Tgl
    FloatAxis(bin_counts[2], axis_ranges[2].first, axis_ranges[2].second, "Snp"),  // snp
    IntAxis(0, bin_counts[3], "sector"),                                           // sector
    IntAxis(0, bin_counts[4], "stackType"),                                        // stack type
    IntAxis(0, bin_counts[5], "charge")                                            // charge type
  );

  // Fill the Boost histogram with the bin contents from the ROOT histogram
  int total_bins = hist->GetNbins();
  for (int bin = 0; bin < total_bins; ++bin) {
    std::vector<int> bin_indices(n_dim);
    double content = hist->GetBinContent(bin, bin_indices.data()); // Get bin coordinates

    // Check if any coordinate is in the underflow (0) or overflow (nbins+1) bins
    bool is_underflow_or_overflow = false;
    for (int dim = 0; dim < n_dim; ++dim) {
      if ((bin_indices[dim] == 0) || (bin_indices[dim] == bin_counts[dim] + 1)) {
        is_underflow_or_overflow = true;
        break;
      }
    }

    // Skip underflow/overflow bins
    if (is_underflow_or_overflow) {
      continue;
    }

    // Assign the content to the corresponding bin in the Boost histogram
    mHist.at(bin_indices[0] - 1, bin_indices[1] - 1, bin_indices[2] - 1, bin_indices[3] - 1, bin_indices[4] - 1, bin_indices[5] - 1) = content;
  }
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

void CalibdEdx::enableDebugOutput(std::string_view fileName)
{
  mDebugOutputStreamer = std::make_unique<o2::utils::TreeStreamRedirector>(fileName.data(), "recreate");
}

void CalibdEdx::disableDebugOutput()
{
  // This will call the TreeStream destructor and write any stored data.
  mDebugOutputStreamer.reset();
}

void CalibdEdx::finalizeDebugOutput() const
{
  if (mDebugOutputStreamer) {
    LOGP(info, "Closing dump file");
    mDebugOutputStreamer->Close();
  }
}

void CalibdEdx::dumpToFile(const char* outFile)
{
  TFile f(outFile, "RECREATE");
  f.WriteObject(this, "calib");
  const auto* thn = getRootHist();
  f.WriteObject(thn, "histogram_data");
}

CalibdEdx CalibdEdx::readFromFile(const char* inFile)
{
  std::unique_ptr<TFile> f(TFile::Open(inFile));
  if (!f || f->IsZombie()) {
    LOGP(error, "Could not open file: {}", inFile);
    CalibdEdx calTmp;
    return calTmp;
  }

  auto obj = f->Get<CalibdEdx>("calib");
  if (!obj) {
    LOGP(error, "Could not read CalibdEdx object from file: {}", inFile);
    CalibdEdx calTmp;
    return calTmp;
  }

  THnF* hTmp = f->Get<THnF>("histogram_data");

  CalibdEdx cal(*obj);
  delete obj;

  if (!hTmp) {
    LOGP(warning, "Could not read histogram from file: {}. Returning empty histogram", inFile);
    return cal;
  }
  cal.setFromRootHist(hTmp);
  return cal;
}

void CalibdEdx::setSigmaFitRange(const float lowerSigma, const float upperSigma)
{
  mSigmaUpper = upperSigma;
  mSigmaLower = lowerSigma;
}
