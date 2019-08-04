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
#include "CommonDataFormat/EvIndex.h"

namespace o2
{
namespace dataformats
{
class MatchInfoTOF
{
  using evIdx = o2::dataformats::EvIndex<int, int>;

 public:
  MatchInfoTOF(evIdx evIdxTOFCl, float chi2, o2::track::TrackLTIntegral trkIntLT, evIdx evIdxTrack = evIdx(0, 0)) : mEvIdxTOFCl(evIdxTOFCl), mChi2(chi2), mIntLT(trkIntLT), mEvIdxTrack(evIdxTrack){};
  MatchInfoTOF() = default;
  void setEvIdxTOFCl(evIdx index) { mEvIdxTOFCl = index; }
  void setEvIdxTrack(evIdx index) { mEvIdxTrack = index; }
  evIdx getEvIdxTOFCl() const { return mEvIdxTOFCl; }
  evIdx getEvIdxTrack() const { return mEvIdxTrack; }
  int getEventTOFClIndex() const { return mEvIdxTOFCl.getEvent(); }
  int getTOFClIndex() const { return mEvIdxTOFCl.getIndex(); }
  int getEventTrackIndex() const { return mEvIdxTrack.getEvent(); }
  int getTrackIndex() const { return mEvIdxTrack.getIndex(); }

  void setChi2(int chi2) { mChi2 = chi2; }
  float getChi2() const { return mChi2; }

  o2::track::TrackLTIntegral& getLTIntegralOut() { return mIntLT; }
  const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mIntLT; }

 private:
  float mChi2;                       // chi2 of the pair track-TOFcluster
  o2::track::TrackLTIntegral mIntLT; ///< L,TOF integral calculated during the propagation
  evIdx mEvIdxTOFCl;                 ///< EvIdx for TOF cluster (first: ev index; second: cluster index)
  evIdx mEvIdxTrack;                 ///< EvIdx for track (first: ev index; second: cluster index)

  ClassDefNV(MatchInfoTOF, 1);
};
} // namespace dataformats
} // namespace o2
#endif
