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

/// \file MatchInfoHMP.h
/// \brief Class to store the output of the matching to HMPID

#ifndef ALICEO2_MATCHINFOHMP_H
#define ALICEO2_MATCHINFOHMP_H

#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/EvIndex.h"

namespace o2
{
namespace dataformats
{
class MatchInfoHMP
{
  using GTrackID = o2::dataformats::GlobalTrackID;

 public:
  MatchInfoHMP(int idxHMPClus, GTrackID idxTrack, float angle = 0, float q = 0, float size = 0, int idxPhotClus = 0) : mIdxHMPClus(idxHMPClus), mIdxTrack(idxTrack), mCkovAngle(angle), mMipCluQ(q), mMipCluSize(size), mIdxPhotClus(idxPhotClus){};
  MatchInfoHMP() = default;

  void setIdxHMPClus(int index) { mIdxHMPClus = index; }
  int getIdxHMPClus() const { return mIdxHMPClus; }

  void setIdxTrack(GTrackID index) { mIdxTrack = index; }
  GTrackID getTrackRef() const { return mIdxTrack; }

  int getTrackIndex() const { return mIdxTrack.getIndex(); }

  void setCkovAngle(float angle) { mCkovAngle = angle; }
  float getCkovAngle() const { return mCkovAngle; }

  void setMipClusQ(float q) { mMipCluQ = q; }
  float getMipClusQ() const { return mMipCluQ; }

  void setMipClusSize(int size) { mMipCluSize = size; }
  int getMipClusSize() const { return mMipCluSize; }

  void setNPhots(int n) { mNPhots = n; }
  int getNPhots() const { return mNPhots; }

  void setPhotIndex(int idx) { mIdxPhotClus = idx; }
  int getPhotIndex() const { return mIdxPhotClus; }

  void print() const;

 private:
  int mIdxHMPClus;       // Idx for HMP cluster
  GTrackID mIdxTrack;    // Idx for track
  float mCkovAngle;      // emission angle value
  float mMipCluQ = 0.0;  // MIP cluster charge
  int mMipCluSize = 0.0; // MIP cluster size
  int mNPhots = 0.0;     // number of candidate photo-electrons
  int mIdxPhotClus;      // index of the first photo

  ClassDefNV(MatchInfoHMP, 1);
};
} // namespace dataformats
} // namespace o2
#endif
