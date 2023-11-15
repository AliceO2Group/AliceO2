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

/// \file LQND.h
/// \brief This file provides the interface for loglikehood policies
/// \author Felix Schlepper

#ifndef O2_TRD_LQND_H
#define O2_TRD_LQND_H

#include "TGraph.h"
#include "TRDPID/PIDBase.h"
#include "DataFormatsTRD/PID.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/Logger.h"
#include "ReconstructionDataFormats/TrackParametrization.h"

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <numeric>

namespace o2
{
namespace trd
{
namespace detail
{
/// Lookup Table class for ccdb upload
template <int nDim>
class LUT
{
 public:
  LUT() = default;
  LUT(std::vector<float> p, std::vector<TGraph> l) : mIntervalsP(p), mLUTs(l) {}

  //
  const TGraph& get(float p, bool isNegative, int iDim = 0) const
  {
    auto upper = std::upper_bound(mIntervalsP.begin(), mIntervalsP.end(), p);
    if (upper == mIntervalsP.end()) {
      // outside of momentum intervals, should not happen
      return mLUTs[0];
    }
    auto index = std::distance(mIntervalsP.begin(), upper);
    index += (isNegative) ? 0 : mIntervalsP.size() * nDim;
    return mLUTs[index + iDim];
  }

 private:
  std::vector<float> mIntervalsP; ///< half-open interval upper bounds starting at 0, e.g., for {1.0,2.0,...} is (-inf,1.0], (1.0,2.0], (2.0, ...)
  std::vector<TGraph> mLUTs;      ///< corresponding likelihood lookup tables

  ClassDefNV(LUT, 1);
};
} // namespace detail

/// This is the ML Base class which defines the interface all machine learning
/// models.
template <int nDim>
class LQND : public PIDBase
{
  static_assert(nDim == 1 || nDim == 2 || nDim == 3, "Likelihood only for 1/2/3 dimension");
  using PIDBase::PIDBase;

 public:
  ~LQND() = default;

  void init(o2::framework::ProcessingContext& pc) final
  {
    // retrieve lookup tables (LUTs) from ccdb
    mLUTs = *(pc.inputs().get<detail::LUT<nDim>*>(Form("lq%ddlut", nDim)));
  }

  float process(const TrackTRD& trkIn, const o2::globaltracking::RecoContainer& input, bool isTPCTRD) const final
  {
    const auto& trkSeed = isTPCTRD ? input.getTPCTracks()[trkIn.getRefGlobalTrackId()].getParamOut() : input.getTPCITSTracks()[trkIn.getRefGlobalTrackId()].getParamOut(); // seeding track
    auto trk = trkSeed;

    const auto isNegative = std::signbit(trkSeed.getSign()); // positive and negative charged particles are treated differently since ExB effects the charge distributions
    const auto& trackletsRaw = input.getTRDTracklets();
    float lei0{1.f}, lei1{1.f}, lei2{1.f}, lpi0{1.f}, lpi1{1.f}, lpi2{1.f}; // likelihood per layer
    for (int iLayer = 0; iLayer < constants::NLAYER; ++iLayer) {
      int trkltId = trkIn.getTrackletIndex(iLayer);
      if (trkltId < 0) { // no tracklet attached
        continue;
      }
      const auto xCalib = input.getTRDCalibratedTracklets()[trkIn.getTrackletIndex(iLayer)].getX();
      auto bz = o2::base::Propagator::Instance()->getNominalBz();
      const auto tgl = trk.getTgl();
      const auto snp = trk.getSnpAt(o2::math_utils::sector2Angle(HelperMethods::getSector(input.getTRDTracklets()[trkIn.getTrackletIndex(iLayer)].getDetector())), xCalib, bz);
      const auto& trklt = trackletsRaw[trkltId];
      const auto [q0, q1, q2] = getCharges(trklt, iLayer, trkIn, input, snp, tgl); // correct charges
      if constexpr (nDim == 1) {
        auto lut = mLUTs.get(trk.getP(), isNegative);
        auto ll1{1.f};
        ll1 = lut.Eval(q0 + q1 + q2);
        lei0 *= ll1;
        lpi0 *= (1.f - ll1);
      } else if (nDim == 2) {
        auto lut1 = mLUTs.get(trk.getP(), isNegative, 0);
        auto lut2 = mLUTs.get(trk.getP(), isNegative, 1);
        auto ll1{1.f};
        auto ll2{1.f};
        ll1 = lut1.Eval(q0 + q2);
        ll2 = lut2.Eval(q1);
        lei0 *= ll1;
        lei1 *= ll2;
        lpi0 *= (1.f - ll1);
        lpi1 *= (1.f - ll2);
      } else {
        auto lut1 = mLUTs.get(trk.getP(), isNegative, 0);
        auto lut2 = mLUTs.get(trk.getP(), isNegative, 1);
        auto lut3 = mLUTs.get(trk.getP(), isNegative, 2);
        auto ll1{1.f};
        auto ll2{1.f};
        auto ll3{1.f};
        ll1 = lut1.Eval(q0);
        ll2 = lut2.Eval(q1);
        ll3 = lut3.Eval(q2);
        lei0 *= ll1;
        lei1 *= ll2;
        lei2 *= ll3;
        lpi0 *= (1.f - ll1);
        lpi1 *= (1.f - ll2);
        lpi2 *= (1.f - ll3);
      }
    }

    return (lei0 * lei1 * lei2) / (lei0 * lei1 * lei2 + lpi0 * lpi1 * lpi2); // combined likelihood
  }

 private:
  detail::LUT<nDim> mLUTs; ///< likelihood lookup tables

  ClassDefNV(LQND, 1);
};

using LQ1D = LQND<1>;
using LQ2D = LQND<2>;
using LQ3D = LQND<3>;

} // namespace trd
} // namespace o2

#endif
