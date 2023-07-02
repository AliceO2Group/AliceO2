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
  MatchInfoHMP(int idxHMPClus, GTrackID idxTrack, float xmip = 0, float ymip = 0, float xtrk = 0, float ytrk = 0, float theta = 0, float phi = 0, float angle = 0, float size = 0, int idxPhotClus = 0, int hmpqn = 0) : mIdxHMPClus(idxHMPClus), mIdxTrack(idxTrack), mCkovAngle(angle), mMipX(xmip), mMipY(ymip), mTrkX(xtrk), mTrkY(ytrk), mTrkTheta(theta), mTrkPhi(phi), mMipCluSize(size), mIdxPhotClus(idxPhotClus), mHMPqn(hmpqn)
  { // Initialize the mPhotCharge array
    for (int i = 0; i < 10; i++) {
      mPhotCharge[i] = 0.0;
    }
  };
  MatchInfoHMP() = default;

  void setIdxHMPClus(int ch, int idx) { mIdxHMPClus = ch * 1000000 + idx; }
  int getIdxHMPClus() const { return mIdxHMPClus; }

  void setIdxTrack(GTrackID index) { mIdxTrack = index; }
  int getTrackIndex() const { return mIdxTrack.getIndex(); }

  GTrackID getTrackRef() const { return mIdxTrack; }

  void setMipX(float x) { mMipX = x; }
  float getMipX() const { return mMipX; }

  void setMipY(float y) { mMipY = y; }
  float getMipY() const { return mMipY; }

  void setTrkX(float x) { mTrkX = x; }
  float getTrkX() const { return mTrkX; }

  void setTrkY(float y) { mTrkY = y; }
  float getTrkY() const { return mTrkY; }

  void setHMPsignal(float angle) { mCkovAngle = angle; }
  float getHMPsignal() const
  {
    if (mCkovAngle > 0) {
      return mCkovAngle - (Int_t)mCkovAngle;
    } else {
      return mCkovAngle;
    }
  }

  void setMipClusSize(int size) { mMipCluSize = size; }
  int getMipClusSize() const { return mMipCluSize; }

  void setNPhots(int n) { mNPhots = n; }
  int getNPhots() const { return mNPhots; }

  void setPhotIndex(int idx) { mIdxPhotClus = idx; }
  int getPhotIndex() const { return mIdxPhotClus; }

  float getOccupancy() const { return (Int_t)mCkovAngle / 10.0; }

  void setHMPIDtrk(float x, float y, float th, float ph)
  {
    mTrkX = x;
    mTrkY = y;
    mTrkTheta = th;
    mTrkPhi = ph;
  }

  void getHMPIDtrk(float& x, float& y, float& th, float& ph) const
  {
    x = mTrkX;
    y = mTrkY;
    th = mTrkTheta;
    ph = mTrkPhi;
  }

  void setHMPIDmip(float x, float y, int q, int nph = 0)
  {
    mMipX = x;
    mMipY = y;
    mHMPqn = 1000000 * nph + q;
  }

  void getHMPIDmip(float& x, float& y, int& q, int& nph) const
  {
    x = mMipX;
    y = mMipY;
    q = mHMPqn % 1000000;
    nph = mHMPqn / 1000000;
  }

  void setPhotCharge(const float* chargeArray)
  {
    for (int i = 0; i < 10; i++) {
      mPhotCharge[i] = chargeArray[i];
    }
  }

  float* getPhotCharge() { return mPhotCharge; }

  void print() const;

 private:
  int mIdxHMPClus;       // Idx for HMP cluster
  GTrackID mIdxTrack;    // Idx for track
  float mMipX;           // local x coordinate of macthed cluster
  float mMipY;           // local y coordinate of macthed cluster
  float mTrkX;           // local x coordinate of extrapolated track intersection point
  float mTrkY;           // local y coordinate of extrapolated track intersection point
  float mTrkTheta;       // theta track
  float mTrkPhi;         // phi track
  float mCkovAngle;      // emission angle value
  int mMipCluSize = 0.0; // MIP cluster size
  int mNPhots = 0.0;     // number of candidate photo-electrons
  int mIdxPhotClus;      // index of the first photo
  int mHMPqn;            // 1000000*number of photon clusters + QDC
  float mPhotCharge[10] = {};

  ClassDefNV(MatchInfoHMP, 2);
};
} // namespace dataformats
} // namespace o2
#endif
