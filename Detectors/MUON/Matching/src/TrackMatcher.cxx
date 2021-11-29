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

/// \file TrackMatcher.cxx
/// \brief Implementation of a class to match MCH and MID tracks
///
/// \author Philippe Pillot, Subatech

#include <algorithm>
#include <map>

#include <Math/SMatrix.h>
#include <Math/SVector.h>

#include "Framework/Logger.h"
#include "MUONMatching/TrackMatcher.h"
#include "MUONMatching/TrackMatcherParam.h"

namespace o2
{
namespace muon
{

using SMatrixSym4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepSym<double, 4>>;
using SMatrix4 = ROOT::Math::SMatrix<double, 4, 4, ROOT::Math::MatRepStd<double, 4>>;
using SVector4 = ROOT::Math::SVector<double, 4>;

//_________________________________________________________________________________________________
/// prepare to run the matching algorithm
void TrackMatcher::init()
{
  // set the maximum chi2 used for matching (4 parameters matched)
  const auto& trackMatcherParam = TrackMatcherParam::Instance();
  mMaxChi2 = 4. * trackMatcherParam.sigmaCut * trackMatcherParam.sigmaCut;
}

//_________________________________________________________________________________________________
/// run the matching algorithm
void TrackMatcher::match(gsl::span<const mch::ROFRecord>& mchROFs, gsl::span<const mch::TrackMCH>& mchTracks,
                         gsl::span<const mid::ROFRecord>& midROFs, gsl::span<const mid::Track>& midTracks)
{
  mMuons.clear();

  if (mchROFs.empty() || midROFs.empty()) {
    return;
  }

  // sort the MID ROFs in increasing time
  std::map<InteractionRecord, int> midSortedROFs{};
  for (int i = 0; i < midROFs.size(); ++i) {
    midSortedROFs[midROFs[i].interactionRecord] = i;
  }

  for (const auto& mchROF : mchROFs) {

    // find the MID ROFs in time with the MCH ROF
    auto itStartMIDROF = midSortedROFs.lower_bound(mchROF.getBCData());
    auto itEndMIDROF = midSortedROFs.upper_bound(mchROF.getBCData() + (mchROF.getBCWidth() - 1));

    for (auto iMCHTrack = mchROF.getFirstIdx(); iMCHTrack <= mchROF.getLastIdx(); ++iMCHTrack) {

      double bestMatchChi2(mMaxChi2);
      int iBestMIDROF(-1);
      uint32_t iBestMIDTrack(0);

      for (auto itMIDROF = itStartMIDROF; itMIDROF != itEndMIDROF; ++itMIDROF) {

        const auto& midROF = midROFs[itMIDROF->second];
        for (auto iMIDTrack = midROF.firstEntry; iMIDTrack < midROF.firstEntry + midROF.nEntries; ++iMIDTrack) {

          // try to match the current MCH track with the current MID track and keep the best matching
          double matchChi2 = match(mchTracks[iMCHTrack], midTracks[iMIDTrack]);
          if (matchChi2 < bestMatchChi2) {
            bestMatchChi2 = matchChi2;
            iBestMIDROF = itMIDROF->second;
            iBestMIDTrack = uint32_t(iMIDTrack);
          }
        }
      }

      // store the muon track if the matching succeeded
      if (iBestMIDROF >= 0) {
        mMuons.emplace_back(uint32_t(iMCHTrack), iBestMIDTrack, midROFs[iBestMIDROF].interactionRecord,
                            bestMatchChi2 / 4.);
      }
    }
  }

  // sort the MUON tracks in increasing BC time
  std::stable_sort(mMuons.begin(), mMuons.end(), [](const TrackMCHMID& mu1, const TrackMCHMID& mu2) {
    return mu1.getIR() < mu2.getIR();
  });
}

//_________________________________________________________________________________________________
/// compute the matching chi2/ndf between these MCH and MID tracks
double TrackMatcher::match(const mch::TrackMCH& mchTrack, const mid::Track& midTrack)
{
  // compute the (X, slopeX, Y, slopeY) parameters difference between the 2 tracks at the z-position of the MID track
  const double* mchParam = mchTrack.getParametersAtMID();
  double dZ = midTrack.getPositionZ() - mchTrack.getZAtMID();
  SVector4 paramDiff(mchParam[0] + mchParam[1] * dZ - midTrack.getPositionX(),
                     mchParam[1] - midTrack.getDirectionX(),
                     mchParam[2] + mchParam[3] * dZ - midTrack.getPositionY(),
                     mchParam[3] - midTrack.getDirectionY());

  // propagate the MCH track covariances to the z-position of the MID track
  SMatrixSym4 mchCov(mchTrack.getCovariancesAtMID(), 10);
  SMatrix4 jacobian(ROOT::Math::SMatrixIdentity{});
  jacobian(0, 1) = dZ;
  jacobian(2, 3) = dZ;
  auto sumCov = ROOT::Math::Similarity(jacobian, mchCov);

  // add the MID track covariances
  sumCov(0, 0) += midTrack.getCovarianceParameter(mid::Track::CovarianceParamIndex::VarX);
  sumCov(1, 0) += midTrack.getCovarianceParameter(mid::Track::CovarianceParamIndex::CovXSlopeX);
  sumCov(1, 1) += midTrack.getCovarianceParameter(mid::Track::CovarianceParamIndex::VarSlopeX);
  sumCov(2, 2) += midTrack.getCovarianceParameter(mid::Track::CovarianceParamIndex::VarY);
  sumCov(3, 2) += midTrack.getCovarianceParameter(mid::Track::CovarianceParamIndex::CovYSlopeY);
  sumCov(3, 3) += midTrack.getCovarianceParameter(mid::Track::CovarianceParamIndex::VarSlopeY);

  // compute the chi2
  if (!sumCov.Invert()) {
    LOG(error) << "Covariance matrix inversion failed: " << sumCov;
    return mMaxChi2;
  }
  return ROOT::Math::Similarity(paramDiff, sumCov);
}

} // namespace muon
} // namespace o2
