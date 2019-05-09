// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatchInfoTOF.h
/// \brief Class to store the output of the matching to TOF

#ifndef ALICEO2_MATCHINFOTOF_H
#define ALICEO2_MATCHINFOTOF_H

#include "ReconstructionDataFormats/TrackLTIntegral.h"

namespace o2
{
namespace dataformats
{
class MatchInfoTOF
{
 public:
  MatchInfoTOF(int indexTOFCl, float chi2, o2::track::TrackLTIntegral trkIntLT) : mTOFClIndex(indexTOFCl), mChi2(chi2), mIntLT(trkIntLT){};
  MatchInfoTOF() = default;
  void setTOFClIndex(int index) { mTOFClIndex = index; }
  int getTOFClIndex() const { return mTOFClIndex; }

  void setChi2(int chi2) { mChi2 = chi2; }
  float getChi2() const { return mChi2; }

  o2::track::TrackLTIntegral& getLTIntegralOut() { return mIntLT; }
  const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mIntLT; }

 private:
  int mTOFClIndex; // index of the TOF cluster used for the matching
  float mChi2;     // chi2 of the pair track-TOFcluster
  o2::track::TrackLTIntegral mIntLT; ///< L,TOF integral calculated during the propagation

  //  ClassDefNV(MatchInfoTOF, 1);
};
}
}
#endif
