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
  MatchInfoTOF(int idLocal, int idxTOFCl, double time, float chi2, o2::track::TrackLTIntegral trkIntLT, GTrackID idxTrack, float dt = 0, float z = 0, float dx = 0, float dz = 0, float dy = 0, float geanttime = 0.0, double t0 = 0.0) : mIdLocal(idLocal), mIdxTOFCl(idxTOFCl), mSignal(time), mChi2(chi2), mIntLT(trkIntLT), mIdxTrack(idxTrack), mDeltaT(dt), mZatTOF(z), mDXatTOF(dx), mDZatTOF(dz), mDYatTOF(dy), mTgeant(geanttime), mT0true(t0){};
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

  void setHitPatternUpDown(bool v) { mHitUpDown = v; }
  bool getHitPatternUpDown() const { return mHitUpDown; }

  void setHitPatternLeftRight(bool v) { mHitLeftRight = v; }
  bool getHitPatternLeftRight() const { return mHitLeftRight; }

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
  void setDYatTOF(float val) { mDYatTOF = val; }
  float getDYatTOF() const { return mDYatTOF; }
  void setSignal(double time) { mSignal = time; }
  double getSignal() const { return mSignal; }

  int getIdLocal() const { return mIdLocal; }

  float getVz() const { return mVz; }
  void setVz(float val) { mVz = val; }
  int getChannel() const { return mChannel; }
  void setChannel(int val) { mChannel = val; }
  float getTgeant() const { return mTgeant; }
  void setTgeant(float val) { mTgeant = val; }
  double getT0true() const { return mT0true; }
  void setT0true(double val) { mT0true = val; }

  enum QualityFlags { isMultiHitX = 0x1 << 0,
                      isMultiHitZ = 0x1 << 1,
                      badDy = 0x1 << 2,
                      isMultiStrip = 0x1 << 3,
                      isNotInPad = 0x1 << 4,
                      chiGT3 = 0x1 << 5,
                      chiGT5 = 0x1 << 6,
                      hasT0sameBC = 0x1 << 7,
                      hasT0_1BCbefore = 0x1 << 8,
                      hasT0_2BCbefore = 0x1 << 9 };

  void setFT0Best(double val, float res = 200.)
  {
    mFT0Best = val;
    mFT0BestRes = res;
  }
  double getFT0Best() const { return mFT0Best; }
  float getFT0BestRes() const { return mFT0BestRes; }

 private:
  int mIdLocal;                      // track id in sector of the pair track-TOFcluster
  float mChi2;                       // chi2 of the pair track-TOFcluster
  o2::track::TrackLTIntegral mIntLT; ///< L,TOF integral calculated during the propagation
  int mIdxTOFCl;                     ///< Idx for TOF cluster
  GTrackID mIdxTrack;                ///< Idx for track
  float mZatTOF = 0.0;               ///< Z position at  TOF
  float mDXatTOF = 0.0;              ///< DX position at  TOF
  float mDZatTOF = 0.0;              ///< DZ position at  TOF
  float mDYatTOF = 0.0;              ///< DY position at  TOF
  float mDeltaT = 0.0;               ///< tTOF - TPC (microsec)
  double mSignal = 0.0;              ///< TOF time in ps
  float mVz = 0.0;                   ///< Vz from TOF match
  int mChannel = -1;                 ///< channel
  // Hit pattern information
  bool mHitUpDown = false;    ///< hit pattern in TOF up-down
  bool mHitLeftRight = false; ///< hit pattern in TOF left-right
  float mTgeant = 0.0;        ///< geant time in MC
  double mT0true = 0.0;       ///< t0true

  double mFT0Best = 0.0;     //< best info for collision time
  float mFT0BestRes = 200.0; //< resolution (in ps) of the best info for collision time

  ClassDefNV(MatchInfoTOF, 9);
};
} // namespace dataformats
} // namespace o2
#endif
