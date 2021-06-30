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

/// \file MatchInfoTOF.h
/// \brief Class to store the output of the matching to TOF

#ifndef ALICEO2_MATCHINFOTOF_H
#define ALICEO2_MATCHINFOTOF_H

#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/EvIndex.h"

namespace o2
{
namespace dataformats
{
class MatchInfoTOF
{
  using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
  using evIdx = o2::dataformats::EvIndex<int, int>;

 public:
  MatchInfoTOF(evIdx evIdxTOFCl, float chi2, o2::track::TrackLTIntegral trkIntLT, evGIdx evIdxTrack, float dt = 0, float z = 0) : mEvIdxTOFCl(evIdxTOFCl), mChi2(chi2), mIntLT(trkIntLT), mEvIdxTrack(evIdxTrack), mDeltaT(dt), mZatTOF(z){};
  MatchInfoTOF() = default;
  void setEvIdxTOFCl(evIdx index) { mEvIdxTOFCl = index; }
  void setEvIdxTrack(evGIdx index) { mEvIdxTrack = index; }
  evIdx getEvIdxTOFCl() const { return mEvIdxTOFCl; }
  evGIdx getEvIdxTrack() const { return mEvIdxTrack; }
  int getEventTOFClIndex() const { return mEvIdxTOFCl.getEvent(); }
  int getTOFClIndex() const { return mEvIdxTOFCl.getIndex(); }
  int getEventTrackIndex() const { return mEvIdxTrack.getEvent(); }
  int getTrackIndex() const { return mEvIdxTrack.getIndex(); }

  void setChi2(int chi2) { mChi2 = chi2; }
  float getChi2() const { return mChi2; }

  o2::track::TrackLTIntegral& getLTIntegralOut() { return mIntLT; }
  const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mIntLT; }
  void print() const;

  void setDeltaT(float val) { mDeltaT = val; }
  float getDeltaT() const { return mDeltaT; }
  void setZatTOF(float val) { mZatTOF = val; }
  float getZatTOF() const { return mZatTOF; }

 private:
  float mChi2;                       // chi2 of the pair track-TOFcluster
  o2::track::TrackLTIntegral mIntLT; ///< L,TOF integral calculated during the propagation
  evIdx mEvIdxTOFCl;                 ///< EvIdx for TOF cluster (first: ev index; second: cluster index)
  evGIdx mEvIdxTrack;                ///< EvIdx for track (first: ev index; second: track global index)
  float mZatTOF = 0.0;               ///< Z position at  TOF
  float mDeltaT = 0.0;               ///< tTOF - TPC (microsec)

  ClassDefNV(MatchInfoTOF, 2);
};
} // namespace dataformats
} // namespace o2
#endif
