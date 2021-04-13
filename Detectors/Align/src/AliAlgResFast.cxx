// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgResFast.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Container for control fast residuals evaluated via derivatives

#include "Align/AliAlgResFast.h"
#include "Align/AliAlgTrack.h"
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgSens.h"
#include "Framework/Logger.h"
#include <TString.h>
#include <TMath.h>
#include <stdio.h>

using namespace TMath;

ClassImp(o2::align::AliAlgResFast);

namespace o2
{
namespace align
{

//____________________________________
AliAlgResFast::AliAlgResFast()
  : mNPoints(0), mNMatSol(0), mNBook(0), mChi2(0), mChi2Ini(0), mD0(0), mD1(0), mSig0(0), mSig1(0), mVolID(0), mLabel(0), mSolMat(0), mMatErr(0)
{
  // def c-tor
  for (int i = 5; i--;)
    mTrCorr[i] = 0;
}

//________________________________________________
AliAlgResFast::~AliAlgResFast()
{
  // d-tor
  delete[] mD0;
  delete[] mD1;
  delete[] mSig0;
  delete[] mSig1;
  delete[] mVolID;
  delete[] mLabel;
  delete[] mSolMat;
  delete[] mMatErr;
}

//________________________________________________
void AliAlgResFast::resize(int np)
{
  // resize container
  if (np > mNBook) {
    delete[] mD0;
    delete[] mD1;
    delete[] mSig0;
    delete[] mSig1;
    delete[] mVolID;
    delete[] mLabel;
    delete[] mSolMat;
    delete[] mMatErr;
    //
    mNBook = 30 + np;
    mD0 = new float[mNBook];
    mD1 = new float[mNBook];
    mSig0 = new float[mNBook];
    mSig1 = new float[mNBook];
    mVolID = new int[mNBook];
    mLabel = new int[mNBook];
    mSolMat = new float[mNBook * 4]; // at most 4 material params per point
    mMatErr = new float[mNBook * 4]; // at most 4 material params per point
    //
    memset(mD0, 0, mNBook * sizeof(float));
    memset(mD1, 0, mNBook * sizeof(float));
    memset(mSig0, 0, mNBook * sizeof(float));
    memset(mSig1, 0, mNBook * sizeof(float));
    memset(mVolID, 0, mNBook * sizeof(int));
    memset(mLabel, 0, mNBook * sizeof(int));
    memset(mSolMat, 0, 4 * mNBook * sizeof(int));
    memset(mMatErr, 0, 4 * mNBook * sizeof(int));
  }
  //
}

//____________________________________________
void AliAlgResFast::Clear(const Option_t*)
{
  // reset record
  mNPoints = 0;
  mNMatSol = 0;
  mTrCorr[4] = 0; // rest will be 100% overwritten
  //
}

//____________________________________________
void AliAlgResFast::Print(const Option_t* /*opt*/) const
{
  // print info
  printf("%3s:%1s (%9s/%5s) %6s | [ %7s:%7s ]\n", "Pnt", "M", "Label",
         "VolID", "Sigma", "resid", "pull/LG");
  for (int irs = 0; irs < mNPoints; irs++) {
    printf("%3d:%1d (%9d/%5d) %6.4f | [%+.2e:%+7.2f]\n",
           irs, 0, mLabel[irs], mVolID[irs], mSig0[irs], mD0[irs],
           mSig0[irs] > 0 ? mD0[irs] / mSig0[irs] : -99);
    printf("%3d:%1d (%9d/%5d) %6.4f | [%+.2e:%+7.2f]\n",
           irs, 1, mLabel[irs], mVolID[irs], mSig1[irs], mD1[irs],
           mSig1[irs] > 0 ? mD1[irs] / mSig1[irs] : -99);
  }
  //
  printf("CorrETP: ");
  for (int i = 0; i < 5; i++)
    printf("%+.3f ", mTrCorr[i]);
  printf("\n");
  printf("MatCorr (corr/sig:pull)\n");
  int nmp = mNMatSol / 4;
  int cnt = 0;
  for (int imp = 0; imp < nmp; imp++) {
    for (int ic = 0; ic < 4; ic++) {
      printf("%+.2e/%.2e:%+8.3f|", mSolMat[cnt], mMatErr[cnt],
             mMatErr[cnt] > 0 ? mSolMat[cnt] / mMatErr[cnt] : -99);
      cnt++;
    }
    printf("\n");
  }
  //
}

//____________________________________________
void AliAlgResFast::setResSigMeas(int ip, int ord, float res, float sig)
{
  // assign residual and error for measurement
  if (ord == 0) {
    mD0[ip] = res;
    mSig0[ip] = sig;
  } else {
    mD1[ip] = res;
    mSig1[ip] = sig;
  }
}

//____________________________________________
void AliAlgResFast::setMatCorr(int id, float res, float sig)
{
  // assign residual and error for material correction
  mSolMat[id] = res;
  mMatErr[id] = sig;
}

//____________________________________________
void AliAlgResFast::setLabel(int ip, int lab, int vol)
{
  // set label/volid of measured volume
  mVolID[ip] = vol;
  mLabel[ip] = lab;
}

} // namespace align
} // namespace o2
