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
  : fNPoints(0), fNMatSol(0), fNBook(0), fChi2(0), fChi2Ini(0), fD0(0), fD1(0), fSig0(0), fSig1(0), fVolID(0), fLabel(0), fSolMat(0), fMatErr(0)
{
  // def c-tor
  for (int i = 5; i--;)
    fTrCorr[i] = 0;
}

//________________________________________________
AliAlgResFast::~AliAlgResFast()
{
  // d-tor
  delete[] fD0;
  delete[] fD1;
  delete[] fSig0;
  delete[] fSig1;
  delete[] fVolID;
  delete[] fLabel;
  delete[] fSolMat;
  delete[] fMatErr;
}

//________________________________________________
void AliAlgResFast::Resize(Int_t np)
{
  // resize container
  if (np > fNBook) {
    delete[] fD0;
    delete[] fD1;
    delete[] fSig0;
    delete[] fSig1;
    delete[] fVolID;
    delete[] fLabel;
    delete[] fSolMat;
    delete[] fMatErr;
    //
    fNBook = 30 + np;
    fD0 = new Float_t[fNBook];
    fD1 = new Float_t[fNBook];
    fSig0 = new Float_t[fNBook];
    fSig1 = new Float_t[fNBook];
    fVolID = new Int_t[fNBook];
    fLabel = new Int_t[fNBook];
    fSolMat = new Float_t[fNBook * 4]; // at most 4 material params per point
    fMatErr = new Float_t[fNBook * 4]; // at most 4 material params per point
    //
    memset(fD0, 0, fNBook * sizeof(Float_t));
    memset(fD1, 0, fNBook * sizeof(Float_t));
    memset(fSig0, 0, fNBook * sizeof(Float_t));
    memset(fSig1, 0, fNBook * sizeof(Float_t));
    memset(fVolID, 0, fNBook * sizeof(Int_t));
    memset(fLabel, 0, fNBook * sizeof(Int_t));
    memset(fSolMat, 0, 4 * fNBook * sizeof(Int_t));
    memset(fMatErr, 0, 4 * fNBook * sizeof(Int_t));
  }
  //
}

//____________________________________________
void AliAlgResFast::Clear(const Option_t*)
{
  // reset record
  fNPoints = 0;
  fNMatSol = 0;
  fTrCorr[4] = 0; // rest will be 100% overwritten
  //
}

//____________________________________________
void AliAlgResFast::Print(const Option_t* /*opt*/) const
{
  // print info
  printf("%3s:%1s (%9s/%5s) %6s | [ %7s:%7s ]\n", "Pnt", "M", "Label",
         "VolID", "Sigma", "resid", "pull/LG");
  for (int irs = 0; irs < fNPoints; irs++) {
    printf("%3d:%1d (%9d/%5d) %6.4f | [%+.2e:%+7.2f]\n",
           irs, 0, fLabel[irs], fVolID[irs], fSig0[irs], fD0[irs],
           fSig0[irs] > 0 ? fD0[irs] / fSig0[irs] : -99);
    printf("%3d:%1d (%9d/%5d) %6.4f | [%+.2e:%+7.2f]\n",
           irs, 1, fLabel[irs], fVolID[irs], fSig1[irs], fD1[irs],
           fSig1[irs] > 0 ? fD1[irs] / fSig1[irs] : -99);
  }
  //
  printf("CorrETP: ");
  for (int i = 0; i < 5; i++)
    printf("%+.3f ", fTrCorr[i]);
  printf("\n");
  printf("MatCorr (corr/sig:pull)\n");
  int nmp = fNMatSol / 4;
  int cnt = 0;
  for (int imp = 0; imp < nmp; imp++) {
    for (int ic = 0; ic < 4; ic++) {
      printf("%+.2e/%.2e:%+8.3f|", fSolMat[cnt], fMatErr[cnt],
             fMatErr[cnt] > 0 ? fSolMat[cnt] / fMatErr[cnt] : -99);
      cnt++;
    }
    printf("\n");
  }
  //
}

//____________________________________________
void AliAlgResFast::SetResSigMeas(int ip, int ord, float res, float sig)
{
  // assign residual and error for measurement
  if (ord == 0) {
    fD0[ip] = res;
    fSig0[ip] = sig;
  } else {
    fD1[ip] = res;
    fSig1[ip] = sig;
  }
}

//____________________________________________
void AliAlgResFast::SetMatCorr(int id, float res, float sig)
{
  // assign residual and error for material correction
  fSolMat[id] = res;
  fMatErr[id] = sig;
}

//____________________________________________
void AliAlgResFast::SetLabel(int ip, Int_t lab, Int_t vol)
{
  // set label/volid of measured volume
  fVolID[ip] = vol;
  fLabel[ip] = lab;
}

} // namespace align
} // namespace o2
