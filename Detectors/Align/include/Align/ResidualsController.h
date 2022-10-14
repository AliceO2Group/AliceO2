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

/// @file   ResidualsController.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Container for control residuals

#ifndef RESIDUALSCONTROLLER_H
#define RESIDUALSCONTROLLER_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include <TMath.h>
#include <Rtypes.h>

namespace o2
{
namespace align
{

class AlignmentTrack;

class ResidualsController
{
 public:
  enum {
    CosmicBit = 0x1,
    VertexBit = 0x1 << 1,
    KalmanBit = 0x1 << 2
  };
  //
  ResidualsController() = default;
  ~ResidualsController();
  //
  int getRun() const { return mRunNumber; }
  void setRun(int r) { mRunNumber = r; }
  uint32_t getFirstTFOrbit() const { return mFirstTFOrbit; }
  void setFirstTFOrbit(uint32_t v) { mFirstTFOrbit = v; }
  o2::dataformats::GlobalTrackID getTrackID() const { return mTrackID; }
  void setTrackID(o2::dataformats::GlobalTrackID t) { mTrackID = t; }
  void setBz(float v) { mBz = v; }
  float getBz() const { return mBz; }

  void setNPoints(int n)
  {
    mNPoints = n;
    resize(n);
  }
  //
  bool isCosmic() const { return testBit(CosmicBit); }
  void setCosmic(bool v = true) { setBit(CosmicBit, v); }

  bool hasVertex() const { return testBit(VertexBit); }
  void setHasVertex(bool v = true) { setBit(VertexBit, v); }
  //
  bool getKalmanDone() const { return testBit(KalmanBit); }
  void setKalmanDone(bool v = true) { setBit(KalmanBit, v); }
  //
  int getNPoints() const { return mNPoints; }
  int getNBook() const { return mNBook; }
  float getChi2() const { return mChi2; }
  float getChi2Ini() const { return mChi2Ini; }
  float getChi2K() const { return mChi2K; }
  float getQ2Pt() const { return mQ2Pt; }
  float getX(int i) const { return mX[i]; }
  float getY(int i) const { return mY[i]; }
  float getZ(int i) const { return mZ[i]; }
  float getSnp(int i) const { return mSnp[i]; }
  float getTgl(int i) const { return mTgl[i]; }
  float getAlpha(int i) const { return mAlpha[i]; }
  float getDY(int i) const { return mDY[i]; }
  float getDZ(int i) const { return mDZ[i]; }
  float getDYK(int i) const { return mDYK[i]; }
  float getDZK(int i) const { return mDZK[i]; }
  //
  float getSigY2K(int i) const { return mSigY2K[i]; }
  float getSigYZK(int i) const { return mSigYZK[i]; }
  float getSigZ2K(int i) const { return mSigZ2K[i]; }
  float getSigmaYK(int i) const { return TMath::Sqrt(mSigY2K[i]); }
  float getSigmaZK(int i) const { return TMath::Sqrt(mSigZ2K[i]); }
  //
  float getSigY2(int i) const { return mSigY2[i]; }
  float getSigYZ(int i) const { return mSigYZ[i]; }
  float getSigZ2(int i) const { return mSigZ2[i]; }
  float getSigmaY(int i) const { return TMath::Sqrt(mSigY2[i]); }
  float getSigmaZ(int i) const { return TMath::Sqrt(mSigZ2[i]); }
  //
  float getSigY2Tot(int i) const { return mSigY2K[i] + mSigY2[i]; }
  float getSigYZTot(int i) const { return mSigYZK[i] + mSigYZ[i]; }
  float getSigZ2Tot(int i) const { return mSigZ2K[i] + mSigZ2[i]; }
  float getSigmaYTot(int i) const { return TMath::Sqrt(getSigY2Tot(i)); }
  float getSigmaZTot(int i) const { return TMath::Sqrt(getSigZ2Tot(i)); }
  //
  int getVolID(int i) const { return mVolID[i]; }
  //
  float getXLab(int i) const;
  float getYLab(int i) const;
  float getZLab(int i) const;
  //
  bool fillTrack(AlignmentTrack& trc, bool doKalman = kTRUE);
  void resize(int n);
  void clear();
  void print(const Option_t* opt = "re") const;
  //
 protected:
  //
  uint16_t mBits = 0;
  int mRunNumber = 0;                        // run
  float mBz = 0;                             // field
  uint32_t mFirstTFOrbit = 0;                // event time stamp
  o2::dataformats::GlobalTrackID mTrackID{}; // track in the event
  int mNPoints = 0;                          // n meas points
  int mNBook = 0;                            //! booked length
  float mChi2 = 0;                           //  chi2 after solution
  float mChi2Ini = 0;                        //  chi2 before solution
  float mChi2K = 0;                          //  chi2 from kalman
  float mQ2Pt = 0;                           //  Q/Pt at reference point
  float* mX = nullptr;                       //[mNPoints] tracking X of cluster
  float* mY = nullptr;                       //[mNPoints] tracking Y of cluster
  float* mZ = nullptr;                       //[mNPoints] tracking Z of cluster
  float* mSnp = nullptr;                     //[mNPoints] track Snp
  float* mTgl = nullptr;                     //[mNPoints] track Tgl
  float* mAlpha = nullptr;                   //[mNPoints] track alpha
  float* mDY = nullptr;                      //[mNPoints] Y residual (track - meas)
  float* mDZ = nullptr;                      //[mNPoints] Z residual (track - meas)
  float* mDYK = nullptr;                     //[mNPoints] Y residual (track - meas) Kalman
  float* mDZK = nullptr;                     //[mNPoints] Z residual (track - meas) Kalman
  float* mSigY2 = nullptr;                   //[mNPoints] Y err^2
  float* mSigYZ = nullptr;                   //[mNPoints] YZ err
  float* mSigZ2 = nullptr;                   //[mNPoints] Z err^2
  float* mSigY2K = nullptr;                  //[mNPoints] Y err^2 of Kalman track smoothing
  float* mSigYZK = nullptr;                  //[mNPoints] YZ err  of Kalman track smoothing
  float* mSigZ2K = nullptr;                  //[mNPoints] Z err^2 of Kalman track smoothing
  int* mVolID = nullptr;                     //[mNPoints] volume id (0 for vertex constraint)
  int* mLabel = nullptr;                     //[mNPoints] label of the volume
  //
  void setBit(uint16_t b, bool v)
  {
    if (v) {
      mBits |= b;
    } else {
      mBits &= ~(b & 0xffff);
    }
  }
  bool testBit(uint16_t b) const
  {
    return mBits & b;
  }

  ClassDefNV(ResidualsController, 1);
};
} // namespace align
} // namespace o2
#endif
