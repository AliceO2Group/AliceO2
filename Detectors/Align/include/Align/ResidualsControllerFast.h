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

/// @file   ResidualsControllerFast.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Container for control fast residuals evaluated via derivatives

#ifndef RESIDUALSCONTROLLERFAST_H
#define RESIDUALSCONTROLLERFAST_H

#include <TObject.h>

namespace o2
{
namespace align
{

class ResidualsControllerFast final : public TObject
{
 public:
  enum { kCosmicBit = BIT(14),
         kVertexBit = BIT(15) };
  //
  ResidualsControllerFast();
  ~ResidualsControllerFast() final;
  //
  void setNPoints(int n)
  {
    mNPoints = n;
    resize(n);
  }
  void setNMatSol(int n) { mNMatSol = n; }
  //
  void setChi2(float v) { mChi2 = v; }
  float getChi2() const { return mChi2; }
  //
  void setChi2Ini(float v) { mChi2Ini = v; }
  float getChi2Ini() const { return mChi2Ini; }
  //
  bool isCosmic() const { return TestBit(kCosmicBit); }
  bool hasVertex() const { return TestBit(kVertexBit); }
  void setCosmic(bool v = true) { SetBit(kCosmicBit, v); }
  void setHasVertex(bool v = true) { SetBit(kVertexBit, v); }
  //
  int getNPoints() const { return mNPoints; }
  int getNMatSol() const { return mNMatSol; }
  int getNBook() const { return mNBook; }
  float getD0(int i) const { return mD0[i]; }
  float getD1(int i) const { return mD1[i]; }
  float getSig0(int i) const { return mSig0[i]; }
  float getSig1(int i) const { return mSig1[i]; }
  int getVolID(int i) const { return mVolID[i]; }
  int getLabel(int i) const { return mLabel[i]; }
  //
  float* getTrCor() const { return (float*)mTrCorr; }
  float* getD0() const { return (float*)mD0; }
  float* getD1() const { return (float*)mD1; }
  float* getSig0() const { return (float*)mSig0; }
  float* getSig1() const { return (float*)mSig1; }
  int* getVolID() const { return (int*)mVolID; }
  int* getLaber() const { return (int*)mLabel; }
  float* getSolMat() const { return (float*)mSolMat; }
  float* getMatErr() const { return (float*)mMatErr; }
  //
  void setResSigMeas(int ip, int ord, float res, float sig);
  void setMatCorr(int id, float res, float sig);
  void setLabel(int ip, int lab, int vol);
  //
  void resize(int n);
  void Clear(const Option_t* opt = "") final;
  void Print(const Option_t* opt = "") const final;
  //
 protected:
  //
  // -------- dummies --------
  ResidualsControllerFast(const ResidualsControllerFast&);
  ResidualsControllerFast& operator=(const ResidualsControllerFast&);
  //
 protected:
  //
  int mNPoints;   // n meas points
  int mNMatSol;   // n local params - ExtTrPar corrections
  int mNBook;     //! booked lenfth
  float mChi2;    // chi2
  float mChi2Ini; // chi2 before local fit
  //
  float mTrCorr[5]; //  correction to ExternalTrackParam
  float* mD0;       //[mNPoints] 1st residual (track - meas)
  float* mD1;       //[mNPoints] 2ns residual (track - meas)
  float* mSig0;     //[mNPoints] ort. error 0
  float* mSig1;     //[mNPoints] ort. errir 1
  int* mVolID;      //[mNPoints] volume id (0 for vertex constraint)
  int* mLabel;      //[mNPoints] label of the volume
  //
  float* mSolMat; //[mNMatSol] // material corrections
  float* mMatErr; //[mNMatSol] // material corrections errors
  //
  ClassDef(ResidualsControllerFast, 1);
};
} // namespace align
} // namespace o2
#endif
