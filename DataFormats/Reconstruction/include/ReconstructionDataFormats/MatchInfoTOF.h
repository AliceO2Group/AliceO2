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
  using GTrackID = o2::dataformats::GlobalTrackID;

 public:
  MatchInfoTOF(int idLocal, int idxTOFCl, double time, float chi2, o2::track::TrackLTIntegral trkIntLT, GTrackID idxTrack, float dt = 0, float z = 0, float dx = 0, float dz = 0) : mIdLocal(idLocal), mIdxTOFCl(idxTOFCl), mSignal(time), mChi2(chi2), mIntLT(trkIntLT), mIdxTrack(idxTrack), mDeltaT(dt), mZatTOF(z), mDXatTOF(dx), mDZatTOF(dz){};
  MatchInfoTOF() = default;
  void setIdxTOFCl(int index) { mIdxTOFCl = index; }
  void setIdxTrack(GTrackID index) { mIdxTrack = index; }
  int getIdxTOFCl() const { return mIdxTOFCl; }
  GTrackID getTrackRef() const { return mIdxTrack; }
  int getEventTOFClIndex() const { return mIdxTOFCl; }
  int getTOFClIndex() const { return mIdxTOFCl; }
  int getTrackIndex() const { return mIdxTrack.getIndex(); }

  void setChi2(float chi2) { mChi2 = chi2; }
  float getChi2() const { return mChi2; }

  void setHitPattern(std::uint8_t hitPattern) { mHitPattern = hitPattern; }
  std::uint8_t getHitPattern() const { return mHitPattern; }

  o2::track::TrackLTIntegral& getLTIntegralOut() { return mIntLT; }
  const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mIntLT; }
  void print() const;

  void setDeltaT(float val) { mDeltaT = val; }
  float getDeltaT() const { return mDeltaT; }
  void setZatTOF(float val) { mZatTOF = val; }
  float getZatTOF() const { return mZatTOF; }
  void setDZatTOF(float val) { mDZatTOF = val; }
  float getDZatTOF() const { return mDZatTOF; }
  void setDXatTOF(float val) { mDXatTOF = val; }
  float getDXatTOF() const { return mDXatTOF; }
  void setSignal(double time) { mSignal = time; }
  double getSignal() const { return mSignal; }

  int getIdLocal() const { return mIdLocal; }

  float getVz() const { return mVz; }
  void setVz(float val) { mVz = val; }
  int getChannel() const { return mChannel; }
  void setChannel(int val) { mChannel = val; }

 private:
  int mIdLocal;                      // track id in sector of the pair track-TOFcluster
  float mChi2;                       // chi2 of the pair track-TOFcluster
  std::uint8_t mHitPattern;          // mask of the hit pattern in TOF
  o2::track::TrackLTIntegral mIntLT; ///< L,TOF integral calculated during the propagation
  int mIdxTOFCl;                     ///< Idx for TOF cluster
  GTrackID mIdxTrack;                ///< Idx for track
  float mZatTOF = 0.0;               ///< Z position at  TOF
  float mDXatTOF = 0.0;              ///< DX position at  TOF
  float mDZatTOF = 0.0;              ///< DZ position at  TOF
  float mDeltaT = 0.0;               ///< tTOF - TPC (microsec)
  double mSignal = 0.0;              ///< TOF time in ps
  float mVz = 0.0;                   ///< Vz from TOF match
  int mChannel = -1;                 ///< channel

  ClassDefNV(MatchInfoTOF, 6);
};
} // namespace dataformats
} // namespace o2
#endif
