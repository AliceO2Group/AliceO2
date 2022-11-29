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

/// @file   Millepede2Record.cxx
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Millepede record in root format (can be converted to proper pede binary format.

#include "Align/Millepede2Record.h"
#include "Align/utils.h"
#include "Align/AlignmentTrack.h"
#include "Framework/Logger.h"
#include <TMath.h>
#include <cstdio>

using namespace TMath;
using namespace o2::align::utils;

namespace o2
{
namespace align
{

//_________________________________________________________
Millepede2Record::~Millepede2Record()
{
  // d-tor
  delete[] mNDLoc;
  delete[] mNDGlo;
  delete[] mVolID;
  delete[] mResid;
  delete[] mMeasType;
  delete[] mResErr;
  delete[] mIDLoc;
  delete[] mIDGlo;
  delete[] mDLoc;
  delete[] mDGlo;
}

//_________________________________________________________
void Millepede2Record::dummyRecord(float res, float err, float dGlo, int labGlo)
{
  // create dummy residuals record
  if (!mNDGlo) {
    resize(1, 1, 1);
  }
  mChi2Ini = 0;
  mNMeas = 1;
  mNResid = 1;
  mNVarLoc = 0;
  mNVarGlo = 1;
  mIDGlo[0] = labGlo;
  mDGlo[0] = dGlo;
  mNDGlo[0] = 1;
  mVolID[0] = -1;
  mResid[0] = res;
  mMeasType[0] = -1;
  mResErr[0] = err;
  //
  mIDLoc[0] = 0;
  mNDLoc[0] = 0;
  mDLoc[0] = 0;
  mNDGloTot = 1;
  mNDLocTot = 0;
  //
}

//_________________________________________________________
bool Millepede2Record::fillTrack(AlignmentTrack& trc, const std::vector<int>& id2Lab)
{
  // fill track info, optionally substitutind glopar par ID by label
  //
  if (!trc.getDerivDone()) {
    LOG(error) << "Track derivatives are not yet evaluated";
    return false;
  }
  mNVarLoc = trc.getNLocPar(); // number of local degrees of freedom in the track
  mNResid = 0;
  mNDLocTot = 0;
  mNDGloTot = 0;
  mChi2Ini = trc.getChi2Ini();
  mQ2Pt = trc.getQ2Pt();
  mTgl = trc.getTgl();
  mNMeas = 0;
  setCosmic(trc.isCosmic());
  // 1) check sizes for buffers, expand if needed
  int np = trc.getNPoints();
  int nres = 0;
  int nlocd = 0;
  int nglod = 0;
  for (int ip = 0; ip < np; ip++) {
    auto pnt = trc.getPoint(ip);
    int ngl = pnt->getNGloDOFs(); // number of DOF's this point depends on
    if (pnt->containsMeasurement()) {
      nres += 2;                    // every point has 2 residuals
      nlocd += mNVarLoc + mNVarLoc; // each residual has max mNVarLoc local derivatives
      nglod += ngl + ngl;           // number of global derivatives
      mNMeas++;
    }
    if (pnt->containsMaterial()) {
      int nmatpar = pnt->getNMatPar();
      nres += nmatpar;  // each point with materials has nmatpar fake residuals
      nlocd += nmatpar; // and nmatpar non-0 local derivatives (orthogonal)
    }
  }
  //
  resize(nres, nlocd, nglod);
  int nParETP = trc.getNLocExtPar(); // numnber of local parameters for reference track param
  //
  const int* gloParID = trc.getGloParID(); // IDs of global DOFs this track depends on
  for (int ip = 0; ip < np; ip++) {
    auto* pnt = trc.getPoint(ip);
    if (pnt->containsMeasurement()) {
      int gloOffs = pnt->getDGloOffs(); // 1st entry of global derivatives for this point
      int nDGlo = pnt->getNGloDOFs();   // number of global derivatives (number of DOFs it depends on)
      if (!pnt->isStatOK()) {
        pnt->incrementStat();
      }
      //
      for (int idim = 0; idim < 2; idim++) { // 2 dimensional orthogonal measurement
        mNDGlo[mNResid] = 0;
        mVolID[mNResid] = pnt->getSensor()->getVolID() + 1;
        //
        // measured residual/error
        mMeasType[mNResid] = idim;
        mResid[mNResid] = trc.getResidual(idim, ip);
        mResErr[mNResid] = Sqrt(pnt->getErrDiag(idim));
        //
        // derivatives over local params
        const double* deriv = trc.getDResDLoc(idim, ip); // array of Dresidual/Dparams_loc
        int nnon0 = 0;
        for (int j = 0; j < nParETP; j++) { // derivatives over reference track parameters
          if (isZeroAbs(deriv[j])) {
            continue;
          }
          nnon0++;
          mDLoc[mNDLocTot] = deriv[j]; // store non-0 derivative
          mIDLoc[mNDLocTot] = j;       // and variable id
          mNDLocTot++;
        }
        int lp0 = pnt->getMinLocVarID();  // point may depend on material variables starting from this one
        int lp1 = pnt->getMaxLocVarID();  // and up to this one (exclusive)
        for (int j = lp0; j < lp1; j++) { // derivatives over material variables
          if (isZeroAbs(deriv[j])) {
            continue;
          }
          nnon0++;
          mDLoc[mNDLocTot] = deriv[j]; // store non-0 derivative
          mIDLoc[mNDLocTot] = j;       // and variable id
          mNDLocTot++;
        }
        //
        mNDLoc[mNResid] = nnon0; // local derivatives done, store their number for this residual
        //
        // derivatives over global params
        nnon0 = 0;
        deriv = trc.getDResDGlo(idim, gloOffs);
        const int* gloIDP = gloParID + gloOffs;
        for (int j = 0; j < nDGlo; j++) {
          if (isZeroAbs(deriv[j])) {
            continue;
          }
          nnon0++;
          mDGlo[mNDGloTot] = deriv[j];                                    // value of derivative
          mIDGlo[mNDGloTot] = id2Lab.empty() ? gloIDP[j] + 1 : id2Lab[gloIDP[j]]; // global DOF ID
          mNDGloTot++;
        }
        mNDGlo[mNResid] = nnon0;
        //
        mNResid++;
      }
    }
    if (pnt->containsMaterial()) {     // material point can add 4 or 5 otrhogonal pseudo-measurements
      int nmatpar = pnt->getNMatPar(); // residuals (correction expectation value)
      //      const float* expMatCorr = pnt->getMatCorrExp(); // expected corrections (diagonalized)
      const float* expMatCov = pnt->getMatCorrCov(); // their diagonalized error matrix
      int offs = pnt->getMaxLocVarID() - nmatpar;    // start of material variables
      // here all derivatives are 1 = dx/dx
      for (int j = 0; j < nmatpar; j++) {
        mMeasType[mNResid] = 10 + j;
        mNDGlo[mNResid] = 0; // mat corrections don't depend on global params
        mVolID[mNResid] = 0; // not associated to global parameter
        mResid[mNResid] = 0; // expectation for MS effects is 0
        mResErr[mNResid] = Sqrt(expMatCov[j]);
        mNDLoc[mNResid] = 1; // only 1 non-0 derivative
        mDLoc[mNDLocTot] = 1.0;
        mIDLoc[mNDLocTot] = offs + j; // variable id
        mNDLocTot++;
        mNResid++;
      }
    }
  }
  //
  if (!mNDGloTot) {
    LOG(info) << "Track does not depend on free global parameters, discard";
    return false;
  }
  return true;
}

//________________________________________________
void Millepede2Record::resize(int nresid, int nloc, int nglo)
{
  // resize container
  if (nresid > mNResidBook) {
    delete[] mMeasType;
    delete[] mNDLoc;
    delete[] mNDGlo;
    delete[] mVolID;
    delete[] mResid;
    delete[] mResErr;
    mMeasType = new int16_t[nresid];
    mNDLoc = new int16_t[nresid];
    mNDGlo = new int[nresid];
    mVolID = new int[nresid];
    mResid = new float[nresid];
    mResErr = new float[nresid];
    mNResidBook = nresid;
    memset(mMeasType, 0, nresid * sizeof(int16_t));
    memset(mNDLoc, 0, nresid * sizeof(int16_t));
    memset(mNDGlo, 0, nresid * sizeof(int));
    memset(mVolID, 0, nresid * sizeof(int));
    memset(mResid, 0, nresid * sizeof(float));
    memset(mResErr, 0, nresid * sizeof(float));
  }
  if (nloc > mNDLocTotBook) {
    delete[] mIDLoc;
    delete[] mDLoc;
    mIDLoc = new int16_t[nloc];
    mDLoc = new float[nloc];
    mNDLocTotBook = nloc;
    memset(mIDLoc, 0, nloc * sizeof(int16_t));
    memset(mDLoc, 0, nloc * sizeof(float));
  }
  if (nglo > mNDGloTotBook) {
    delete[] mIDGlo;
    delete[] mDGlo;
    mIDGlo = new int[nglo];
    mDGlo = new float[nglo];
    mNDGloTotBook = nglo;
    memset(mIDGlo, 0, nglo * sizeof(int));
    memset(mDGlo, 0, nglo * sizeof(float));
  }
  //
}

//____________________________________________
void Millepede2Record::clear()
{
  // reset record
  mBits = 0;
  mNResid = 0;
  mNVarLoc = 0;
  mNVarGlo = 0;
  mNDLocTot = 0;
  mNDGloTot = 0;
}

//____________________________________________
void Millepede2Record::print() const
{
  // print info
  //
  printf("Track %s TForbit:%d Run:%d\n", mTrackID.asString().c_str(), mFirstTFOrbit, mRunNumber);
  printf("Nmeas:%3d Q/pt:%+.2e Tgl:%+.2e Chi2Ini:%.1f\n", mNMeas, mQ2Pt, mTgl, mChi2Ini);
  printf("NRes: %3d NLoc: %3d NGlo:%3d | Stored: Loc:%3d Glo:%5d\n",
         mNResid, mNVarLoc, mNVarGlo, mNDLocTot, mNDGloTot);
  //
  int curLoc = 0, curGlo = 0;
  const int kNColLoc = 5;
  for (int ir = 0; ir < mNResid; ir++) {
    int ndloc = mNDLoc[ir], ndglo = mNDGlo[ir];
    printf("Res:%3d Type:%d %+e (%+e) | NDLoc:%3d NDGlo:%4d [VID:%5d]\n",
           ir, mMeasType[ir], mResid[ir], mResErr[ir], ndloc, ndglo, getVolID(ir));
    //
    printf("Local Derivatives:\n");
    bool eolOK = true;
    for (int id = 0; id < ndloc; id++) {
      int jd = id + curLoc;
      printf("[%3d] %+.2e  ", mIDLoc[jd], mDLoc[jd]);
      if (((id + 1) % kNColLoc) == 0) {
        printf("\n");
        eolOK = true;
      } else {
        eolOK = false;
      }
    }
    if (!eolOK) {
      printf("\n");
    }
    curLoc += ndloc;
    //
    //
    printf("Global Derivatives:\n");
    //    eolOK = true;
    //    const int kNColGlo=6;
    int prvID = -1;
    for (int id = 0; id < ndglo; id++) {
      int jd = id + curGlo;
      //      eolOK==false;
      if (prvID > mIDGlo[jd] % 100) {
        printf("\n"); /* eolOK = true;*/
      }
      printf("[%5d] %+.2e  ", mIDGlo[jd], mDGlo[jd]);
      //      if (((id+1)%kNColGlo)==0)
      //      else eolOK = false;
      prvID = mIDGlo[jd] % 100;
    }
    //    if (!eolOK) printf("\n");
    printf("\n");
    curGlo += ndglo;
    //
  }
}

} // namespace align
} // namespace o2
