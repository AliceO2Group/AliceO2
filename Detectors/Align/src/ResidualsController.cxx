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

#include "Align/ResidualsController.h"
#include "Align/AlignmentTrack.h"
#include "Align/AlignmentPoint.h"
#include "Align/AlignableSensor.h"
#include "Framework/Logger.h"
#include <TString.h>
#include <TMath.h>
#include <cstdio>

using namespace TMath;

namespace o2
{
namespace align
{

//________________________________________________
ResidualsController::~ResidualsController()
{
  // d-tor
  delete[] mX;
  delete[] mY;
  delete[] mZ;
  delete[] mSnp;
  delete[] mTgl;
  delete[] mAlpha;
  delete[] mDY;
  delete[] mDZ;
  delete[] mSigY2;
  delete[] mSigYZ;
  delete[] mSigZ2;
  delete[] mDYK;
  delete[] mDZK;
  delete[] mSigY2K;
  delete[] mSigYZK;
  delete[] mSigZ2K;
  delete[] mVolID;
  delete[] mLabel;
}

//________________________________________________
void ResidualsController::resize(int np)
{
  // resize container
  if (np > mNBook) {
    delete[] mX;
    delete[] mY;
    delete[] mZ;
    delete[] mSnp;
    delete[] mTgl;
    delete[] mAlpha;
    delete[] mDY;
    delete[] mDZ;
    delete[] mSigY2;
    delete[] mSigYZ;
    delete[] mSigZ2;
    delete[] mDYK;
    delete[] mDZK;
    delete[] mSigY2K;
    delete[] mSigYZK;
    delete[] mSigZ2K;
    delete[] mVolID;
    delete[] mLabel;
    //
    mNBook = 100 + np;
    mX = new float[mNBook];
    mY = new float[mNBook];
    mZ = new float[mNBook];
    mSnp = new float[mNBook];
    mTgl = new float[mNBook];
    mAlpha = new float[mNBook];
    mDY = new float[mNBook];
    mDZ = new float[mNBook];
    mSigY2 = new float[mNBook];
    mSigYZ = new float[mNBook];
    mSigZ2 = new float[mNBook];
    mDYK = new float[mNBook];
    mDZK = new float[mNBook];
    mSigY2K = new float[mNBook];
    mSigYZK = new float[mNBook];
    mSigZ2K = new float[mNBook];
    mVolID = new int[mNBook];
    mLabel = new int[mNBook];
    //
    memset(mX, 0, mNBook * sizeof(float));
    memset(mY, 0, mNBook * sizeof(float));
    memset(mZ, 0, mNBook * sizeof(float));
    memset(mSnp, 0, mNBook * sizeof(float));
    memset(mTgl, 0, mNBook * sizeof(float));
    memset(mAlpha, 0, mNBook * sizeof(float));
    memset(mDY, 0, mNBook * sizeof(float));
    memset(mDZ, 0, mNBook * sizeof(float));
    memset(mSigY2, 0, mNBook * sizeof(float));
    memset(mSigYZ, 0, mNBook * sizeof(float));
    memset(mSigZ2, 0, mNBook * sizeof(float));
    memset(mDYK, 0, mNBook * sizeof(float));
    memset(mDZK, 0, mNBook * sizeof(float));
    memset(mSigY2K, 0, mNBook * sizeof(float));
    memset(mSigYZK, 0, mNBook * sizeof(float));
    memset(mSigZ2K, 0, mNBook * sizeof(float));
    memset(mVolID, 0, mNBook * sizeof(int));
    memset(mLabel, 0, mNBook * sizeof(int));
  }
  //
}

//____________________________________________
void ResidualsController::clear()
{
  // reset record
  mBits = 0;
  mNPoints = 0;
  mRunNumber = 0;
  mFirstTFOrbit = 0;
  mChi2 = 0;
  mChi2K = 0;
  mQ2Pt = 0;
  //
}

//____________________________________________
void ResidualsController::print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  bool lab = opts.Contains("l");
  printf("Track %s TForbit:%d Run:%d\n", mTrackID.asString().c_str(), mFirstTFOrbit, mRunNumber);
  printf("%5sTr.", isCosmic() ? "Cosm." : "Coll.");
  // in case of cosmic, other leg should be shown
  printf("Run:%6d Bz:%+4.1f Np: %3d q/Pt:%+.4f | Chi2: Ini: %6.1f LinSol:%6.1f Kalm:%6.1f |Vtx:%3s\n",
         mRunNumber, mBz, mNPoints, mQ2Pt, mChi2Ini, mChi2, mChi2K, hasVertex() ? "ON" : "OFF");
  if (opts.Contains("r")) {
    bool ers = opts.Contains("e");
    printf("%5s %7s %s %7s %7s %7s %5s %5s %9s %9s",
           " VID ", " Label ", " Alp ", "   X   ", "   Y   ", "   Z   ", " Snp ", " Tgl ", "    DY   ", "    DZ   ");
    if (ers) {
      printf(" %8s %8s %8s", " pSgYY ", " pSgYZ ", " pSgZZ ");
    } // cluster errors
    if (getKalmanDone()) {
      printf(" %9s %9s", "    DYK  ", "    DZK  ");
      if (ers) {
        printf(" %8s %8s %8s", " tSgYY ", " tSgYZ ", " tSgZZ ");
      } // track errors
    }
    printf("\n");
    for (int i = 0; i < mNPoints; i++) {
      float x = mX[i], y = mY[i], z = mZ[i];
      if (lab) {
        x = getXLab(i);
        y = getYLab(i);
        z = getZLab(i);
      }
      printf("%5d %7d %+5.2f %+7.2f %+7.2f %+7.2f %+5.2f %+5.2f %+9.2e %+9.2e",
             mVolID[i], mLabel[i], mAlpha[i], x, y, z, mSnp[i], mTgl[i], mDY[i], mDZ[i]);
      if (ers) {
        printf(" %.2e %+.1e %.2e", mSigY2[i], mSigYZ[i], mSigZ2[i]);
      }
      if (getKalmanDone()) {
        printf(" %+9.2e %+9.2e", mDYK[i], mDZK[i]);
        if (ers) {
          printf(" %.2e %+.1e %.2e", mSigY2K[i], mSigYZK[i], mSigZ2K[i]);
        }
      }
      printf("\n");
    }
  }
}

//____________________________________________________________
bool ResidualsController::fillTrack(AlignmentTrack& trc, bool doKalman)
{
  // fill tracks residuals info
  int nps, np = trc.getNPoints();
  if (trc.getInnerPoint()->containsMeasurement()) {
    setHasVertex();
    nps = np;
  } else {
    nps = np - 1;
  } // ref point is dummy?
  if (nps < 0) {
    return true;
  }
  setCosmic(trc.isCosmic());
  //
  setNPoints(nps);
  mQ2Pt = trc.getQ2Pt();
  mChi2 = trc.getChi2();
  mChi2Ini = trc.getChi2Ini();
  int nfill = 0;
  for (int i = 0; i < np; i++) {
    auto pnt = trc.getPoint(i);
    int inv = pnt->isInvDir() ? -1 : 1; // Flag inversion for cosmic upper leg
    if (!pnt->containsMeasurement()) {
      continue;
    }
    if (!pnt->isStatOK()) {
      pnt->incrementStat();
    }
    mVolID[nfill] = pnt->getVolID();
    mLabel[nfill] = pnt->getSensor()->getInternalID();
    mAlpha[nfill] = pnt->getAlphaSens();
    mX[nfill] = pnt->getXTracking();
    mY[nfill] = pnt->getYTracking();
    mZ[nfill] = pnt->getZTracking();
    mDY[nfill] = pnt->getResidY();
    mDZ[nfill] = pnt->getResidZ();
    mSigY2[nfill] = pnt->getYZErrTracking()[0];
    mSigYZ[nfill] = pnt->getYZErrTracking()[1];
    mSigZ2[nfill] = pnt->getYZErrTracking()[2];
    //
    mSnp[nfill] = pnt->getTrParamWSA()[AlignmentPoint::kParSnp];
    mTgl[nfill] = pnt->getTrParamWSA()[AlignmentPoint::kParTgl];
    //
    nfill++;
  }
  if (nfill != nps) {
    trc.Print("p");
    LOG(fatal) << nfill << " residuals were stored instead of " << nps;
  }
  //
  setKalmanDone(false);
  int nfilk = 0;
  if (doKalman && trc.residKalman()) {
    for (int i = 0; i < np; i++) {
      AlignmentPoint* pnt = trc.getPoint(i);
      if (!pnt->containsMeasurement()) {
        continue;
      }
      if (mVolID[nfilk] != int(pnt->getVolID())) {
        LOG(fatal) << "Mismatch in Kalman filling for point " << i << ": filled VID:" << mVolID[nfilk] << ", point VID:" << pnt->getVolID();
      }
      const double* wsA = pnt->getTrParamWSA();
      mDYK[nfilk] = pnt->getResidY();
      mDZK[nfilk] = pnt->getResidZ();
      mSigY2K[nfilk] = wsA[2];
      mSigYZK[nfilk] = wsA[3];
      mSigZ2K[nfilk] = wsA[4];
      //
      nfilk++;
    }
    //
    mChi2K = trc.getChi2();
    setKalmanDone(true);
  }

  return true;
}

//_________________________________________________
float ResidualsController::getXLab(int i) const
{
  // cluster lab X
  return Abs(mX[i]) * Cos(mAlpha[i]) - mY[i] * Sin(mAlpha[i]);
}

//_________________________________________________
float ResidualsController::getYLab(int i) const
{
  // cluster lab Y
  return Abs(mX[i]) * Sin(mAlpha[i]) + mY[i] * Cos(mAlpha[i]);
}

//_________________________________________________
float ResidualsController::getZLab(int i) const
{
  // cluster lab Z
  return mZ[i];
}

} // namespace align
} // namespace o2
