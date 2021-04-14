// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <TObject.h>
#include <TMath.h>

namespace o2
{
namespace align
{

class AlignmentTrack;

class ResidualsController : public TObject
{
 public:
  enum { kCosmicBit = BIT(14),
         kVertexBit = BIT(15),
         kKalmanDoneBit = BIT(16) };
  //
  ResidualsController();
  virtual ~ResidualsController();
  //
  void setRun(int r) { mRun = r; }
  void setBz(float v) { mBz = v; }
  void setTimeStamp(uint32_t v) { mTimeStamp = v; }
  void setTrackID(uint32_t v) { mTrackID = v; }
  void setNPoints(int n)
  {
    mNPoints = n;
    resize(n);
  }
  //
  bool isCosmic() const { return TestBit(kCosmicBit); }
  bool hasVertex() const { return TestBit(kVertexBit); }
  void setCosmic(bool v = kTRUE) { SetBit(kCosmicBit, v); }
  void setHasVertex(bool v = kTRUE) { SetBit(kVertexBit, v); }
  //
  bool getKalmanDone() const { return TestBit(kKalmanDoneBit); }
  void setKalmanDone(bool v = kTRUE) { SetBit(kKalmanDoneBit, v); }
  //
  int getRun() const { return mRun; }
  float getBz() const { return mBz; }
  uint32_t getTimeStamp() const { return mTimeStamp; }
  uint32_t getTrackID() const { return mTrackID; }
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
  bool fillTrack(AlignmentTrack* trc, bool doKalman = kTRUE);
  void resize(int n);
  virtual void Clear(const Option_t* opt = "");
  virtual void Print(const Option_t* opt = "re") const;
  //
 protected:
  //
  // -------- dummies --------
  ResidualsController(const ResidualsController&);
  ResidualsController& operator=(const ResidualsController&);
  //
 protected:
  //
  int mRun;            // run
  float mBz;           // field
  uint32_t mTimeStamp; // event time
  uint32_t mTrackID;   // track ID
  int mNPoints;        // n meas points
  int mNBook;          //! booked lenfth
  float mChi2;         //  chi2 after solution
  float mChi2Ini;      //  chi2 before solution
  float mChi2K;        //  chi2 from kalman
  float mQ2Pt;         //  Q/Pt at reference point
  float* mX;           //[mNPoints] tracking X of cluster
  float* mY;           //[mNPoints] tracking Y of cluster
  float* mZ;           //[mNPoints] tracking Z of cluster
  float* mSnp;         //[mNPoints] track Snp
  float* mTgl;         //[mNPoints] track Tgl
  float* mAlpha;       //[mNPoints] track alpha
  float* mDY;          //[mNPoints] Y residual (track - meas)
  float* mDZ;          //[mNPoints] Z residual (track - meas)
  float* mDYK;         //[mNPoints] Y residual (track - meas) Kalman
  float* mDZK;         //[mNPoints] Z residual (track - meas) Kalman
  float* mSigY2;       //[mNPoints] Y err^2
  float* mSigYZ;       //[mNPoints] YZ err
  float* mSigZ2;       //[mNPoints] Z err^2
  float* mSigY2K;      //[mNPoints] Y err^2 of Kalman track smoothing
  float* mSigYZK;      //[mNPoints] YZ err  of Kalman track smoothing
  float* mSigZ2K;      //[mNPoints] Z err^2 of Kalman track smoothing
  int* mVolID;         //[mNPoints] volume id (0 for vertex constraint)
  int* mLabel;         //[mNPoints] label of the volume
  //
  ClassDef(ResidualsController, 2);
};
} // namespace align
} // namespace o2
#endif
