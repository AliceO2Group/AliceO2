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
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <numeric>

namespace o2
{
namespace trd
{

/// This is the ML Base class which defines the interface all machine learning
/// models.
template <int nDim>
class LQND : public PIDBase
{
  static_assert(nDim == 1 || nDim == 3, "Likelihood only for 1/3 dimension");
  using PIDBase::PIDBase;

 public:
  ~LQND() = default;

  void init(o2::framework::ProcessingContext& pc) final
  {
    // retrieve lookup tables (LUTs) from ccdb
    mLUTs = *(pc.inputs().get<std::vector<TGraph>*>(Form("lq%ddlut", nDim)));
  }

  float process(const TrackTRD& trk, const o2::globaltracking::RecoContainer& input, bool isTPCTRD) const final
  {
    const auto& trkSeed = isTPCTRD ? input.getTPCTracks()[trk.getRefGlobalTrackId()].getParamOut() : input.getTPCITSTracks()[trk.getRefGlobalTrackId()].getParamOut();
    const auto isNegative = std::signbit(trkSeed.getQ2Pt()); // positive and negative charged particles are treated differently since ExB effects the charge distributions
    const auto& trackletsRaw = input.getTRDTracklets();
    float lei0{1.f}, lei1{1.f}, lei2{1.f}, lpi0{1.f}, lpi1{1.f}, lpi2{1.f}; // likelihood per layer
    for (int iLayer = 0; iLayer < constants::NLAYER; ++iLayer) {
      int trkltId = trk.getTrackletIndex(iLayer);
      if (trkltId < 0) { // no tracklet attached
        continue;
      }
      const auto& trklt = trackletsRaw[trkltId];
      const auto [q0, q1, q2] = getCharges(trklt, iLayer, trk, input); // correct charges
      if constexpr (nDim == 1) {
        auto ll1{1.f};
        if (isNegative) { // particle is negativily charged
          ll1 = mLUTs[0].Eval(q0 + q1 + q2);
        } else { // particle is positivily charged
          ll1 = mLUTs[0 + nDim].Eval(q0 + q1 + q2);
        }
        lei0 *= ll1;
        lpi0 *= (1.f - ll1);
      } else {
        auto ll1{1.f};
        auto ll2{1.f};
        auto ll3{1.f};
        if (isNegative) {
          ll1 = mLUTs[0].Eval(q0);
          ll2 = mLUTs[1].Eval(q1);
          ll3 = mLUTs[2].Eval(q2);
        } else {
          ll1 = mLUTs[0 + nDim].Eval(q0);
          ll2 = mLUTs[1 + nDim].Eval(q1);
          ll3 = mLUTs[2 + nDim].Eval(q2);
        }
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
  std::vector<TGraph> mLUTs; ///< likelihood lookup tables, there are 2 * nDim LUTs, first are the one for negativly charged particles then for positively charged ones
};

using LQ1D = LQND<1>;
using LQ3D = LQND<3>;

} // namespace trd
} // namespace o2

#endif
