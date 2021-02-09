/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include "AliAlgRes.h"
#include "AliAlgTrack.h"
#include "AliAlgPoint.h"
#include "AliAlgSens.h"
#include "AliLog.h"
#include <TString.h>
#include <TMath.h>
#include <stdio.h>

using namespace TMath;

ClassImp(AliAlgRes)

//____________________________________
AliAlgRes::AliAlgRes() 
: fRun(0)
  ,fBz(0)
  ,fTimeStamp(0)
  ,fTrackID(0)
  ,fNPoints(0)
  ,fNBook(0)
  ,fChi2(0)
  ,fChi2Ini(0)
  ,fChi2K(0)
  ,fQ2Pt(0)
  ,fX(0)
  ,fY(0)
  ,fZ(0)
  ,fSnp(0)
  ,fTgl(0)
  ,fAlpha(0)
  ,fDY(0)
  ,fDZ(0)
  ,fDYK(0)
  ,fDZK(0)
  ,fSigY2(0)
  ,fSigYZ(0)
  ,fSigZ2(0)
  ,fSigY2K(0)
  ,fSigYZK(0)
  ,fSigZ2K(0)
  ,fVolID(0)
  ,fLabel(0)
{
  // def c-tor
}

//________________________________________________
AliAlgRes::~AliAlgRes()
{
  // d-tor
  delete[] fX;
  delete[] fY;
  delete[] fZ;
  delete[] fSnp;
  delete[] fTgl;
  delete[] fAlpha;
  delete[] fDY;
  delete[] fDZ;
  delete[] fSigY2;
  delete[] fSigYZ;
  delete[] fSigZ2;
  delete[] fDYK;
  delete[] fDZK;
  delete[] fSigY2K;
  delete[] fSigYZK;
  delete[] fSigZ2K;
  delete[] fVolID;
  delete[] fLabel;
}

//________________________________________________
void AliAlgRes::Resize(Int_t np)
{
  // resize container
  if (np>fNBook) {
    delete[] fX;
    delete[] fY;
    delete[] fZ;
    delete[] fSnp;
    delete[] fTgl;
    delete[] fAlpha;
    delete[] fDY;
    delete[] fDZ;
    delete[] fSigY2;
    delete[] fSigYZ;
    delete[] fSigZ2;
    delete[] fDYK;
    delete[] fDZK;
    delete[] fSigY2K;
    delete[] fSigYZK;
    delete[] fSigZ2K;
    delete[] fVolID;
    delete[] fLabel;
    //
    fNBook = 100+np;
    fX       = new Float_t[fNBook];
    fY       = new Float_t[fNBook];
    fZ       = new Float_t[fNBook];
    fSnp     = new Float_t[fNBook];
    fTgl     = new Float_t[fNBook];
    fAlpha   = new Float_t[fNBook];
    fDY      = new Float_t[fNBook];
    fDZ      = new Float_t[fNBook];
    fSigY2   = new Float_t[fNBook];
    fSigYZ   = new Float_t[fNBook];
    fSigZ2   = new Float_t[fNBook];
    fDYK     = new Float_t[fNBook];
    fDZK     = new Float_t[fNBook];
    fSigY2K  = new Float_t[fNBook];
    fSigYZK  = new Float_t[fNBook];
    fSigZ2K  = new Float_t[fNBook];
    fVolID   = new Int_t[fNBook];
    fLabel   = new Int_t[fNBook];
    //
    memset(fX    , 0,fNBook*sizeof(Float_t));
    memset(fY    , 0,fNBook*sizeof(Float_t));
    memset(fZ    , 0,fNBook*sizeof(Float_t));
    memset(fSnp  , 0,fNBook*sizeof(Float_t));
    memset(fTgl  , 0,fNBook*sizeof(Float_t));
    memset(fAlpha, 0,fNBook*sizeof(Float_t));
    memset(fDY   , 0,fNBook*sizeof(Float_t));
    memset(fDZ   , 0,fNBook*sizeof(Float_t));
    memset(fSigY2, 0,fNBook*sizeof(Float_t));
    memset(fSigYZ, 0,fNBook*sizeof(Float_t));
    memset(fSigZ2, 0,fNBook*sizeof(Float_t));
    memset(fDYK  , 0,fNBook*sizeof(Float_t));
    memset(fDZK  , 0,fNBook*sizeof(Float_t));
    memset(fSigY2K,0,fNBook*sizeof(Float_t));
    memset(fSigYZK,0,fNBook*sizeof(Float_t));
    memset(fSigZ2K,0,fNBook*sizeof(Float_t));
    memset(fVolID, 0,fNBook*sizeof(Int_t));
    memset(fLabel, 0,fNBook*sizeof(Int_t));
  }
  //
}

//____________________________________________
void AliAlgRes::Clear(const Option_t *)
{
  // reset record
  TObject::Clear();
  ResetBit(0xffffffff);
  fNPoints = 0;
  fRun = 0;
  fTimeStamp = 0;
  fTrackID = 0;
  fChi2 = 0;
  fChi2K= 0;
  fQ2Pt = 0;
  //
}

//____________________________________________
void AliAlgRes::Print(const Option_t *opt) const
{
  // print info
  TString opts = opt; opts.ToLower();
  Bool_t lab = opts.Contains("l");
  printf("%5sTr.",IsCosmic() ? "Cosm.":"Coll.");
  if (IsCosmic()) printf("%2d/%2d ",fTrackID>>16,fTrackID&0xffff);
  else            printf("%5d ",fTrackID);
  printf("Run:%6d Bz:%+4.1f Np: %3d q/Pt:%+.4f | Chi2: Ini: %6.1f LinSol:%6.1f Kalm:%6.1f |Vtx:%3s| TStamp:%d\n",
	 fRun,fBz,fNPoints,fQ2Pt,fChi2Ini,fChi2,fChi2K,HasVertex() ? "ON":"OFF",fTimeStamp);
  if (opts.Contains("r")) {
    Bool_t ers = opts.Contains("e");
    printf("%5s %7s %s %7s %7s %7s %5s %5s %9s %9s",
	   " VID "," Label "," Alp ","   X   ","   Y   ","   Z   "," Snp "," Tgl ","    DY   ","    DZ   ");
    if (ers) printf(" %8s %8s %8s"," pSgYY "," pSgYZ "," pSgZZ "); // cluster errors
    if (GetKalmanDone()) {
      printf(" %9s %9s","    DYK  ","    DZK  ");
      if (ers) printf(" %8s %8s %8s"," tSgYY "," tSgYZ "," tSgZZ "); // track errors
    }
    printf("\n");
    for (int i=0;i<fNPoints;i++) {
      float x=fX[i],y=fY[i],z=fZ[i];
      if (lab) {
	x = GetXLab(i);
	y = GetYLab(i);
	z = GetZLab(i);
      }
      printf("%5d %7d %+5.2f %+7.2f %+7.2f %+7.2f %+5.2f %+5.2f %+9.2e %+9.2e",
	     fVolID[i],fLabel[i],fAlpha[i],x,y,z,fSnp[i],fTgl[i],fDY[i],fDZ[i]);
      if (ers) printf(" %.2e %+.1e %.2e",fSigY2[i],fSigYZ[i],fSigZ2[i]);
      if (GetKalmanDone()) {
	printf(" %+9.2e %+9.2e",fDYK[i],fDZK[i]);
	if (ers) printf(" %.2e %+.1e %.2e",fSigY2K[i],fSigYZK[i],fSigZ2K[i]);
      }
      printf("\n");
    }
  }
}

//____________________________________________________________
Bool_t AliAlgRes::FillTrack(AliAlgTrack* trc, Bool_t doKalman)
{
  // fill tracks residuals info
  int nps,np = trc->GetNPoints();
  if (trc->GetInnerPoint()->ContainsMeasurement()) {
    SetHasVertex();
    nps = np;
  }
  else nps = np-1; // ref point is dummy?
  if (nps<0) return kTRUE;
  SetCosmic(trc->IsCosmic());
  //  
  SetNPoints(nps);
  fQ2Pt = trc->GetSigned1Pt();
  fChi2 = trc->GetChi2();
  fChi2Ini = trc->GetChi2Ini();
  int nfill = 0;
  for (int i=0;i<np;i++) {
    AliAlgPoint* pnt = trc->GetPoint(i);
    int inv = pnt->IsInvDir() ? -1:1;    // Flag invertion for cosmic upper leg
    if (!pnt->ContainsMeasurement()) continue;
    if (!pnt->IsStatOK()) pnt->IncrementStat();
    fVolID[nfill] = pnt->GetVolID();
    fLabel[nfill] = pnt->GetSensor()->GetInternalID();
    fAlpha[nfill] = pnt->GetAlphaSens();
    fX[nfill]     = pnt->GetXPoint()*inv;
    fY[nfill]     = pnt->GetYTracking();
    fZ[nfill]     = pnt->GetZTracking();
    fDY[nfill]    = pnt->GetResidY();
    fDZ[nfill]    = pnt->GetResidZ();
    fSigY2[nfill] = pnt->GetYZErrTracking()[0];
    fSigYZ[nfill] = pnt->GetYZErrTracking()[1];
    fSigZ2[nfill] = pnt->GetYZErrTracking()[2];
    //
    fSnp[nfill]   = pnt->GetTrParamWSA()[AliAlgPoint::kParSnp];
    fTgl[nfill]   = pnt->GetTrParamWSA()[AliAlgPoint::kParTgl];
    //
    nfill++;
  }
  if (nfill!=nps) {
    trc->Print("p");
    AliFatalF("Something is wrong: %d residuals were stored instead of %d",nfill,nps);
  }
  //
  SetKalmanDone(kFALSE);
  int nfilk=0;
  if (doKalman && trc->ResidKalman()) {
    for (int i=0;i<np;i++) {
      AliAlgPoint* pnt = trc->GetPoint(i);
      if (!pnt->ContainsMeasurement()) continue;
      if (fVolID[nfilk] != int(pnt->GetVolID())) {
	AliFatalF("Mismatch in Kalman filling for point %d: filled VID:%d, point VID:%d",
		  i,fVolID[nfilk],pnt->GetVolID());
      }
      const double* wsA = pnt->GetTrParamWSA();
      fDYK[nfilk]    = pnt->GetResidY();
      fDZK[nfilk]    = pnt->GetResidZ();
      fSigY2K[nfilk] = wsA[2];
      fSigYZK[nfilk] = wsA[3];
      fSigZ2K[nfilk] = wsA[4];
      //
      nfilk++;
    }
    //
    fChi2K = trc->GetChi2();
    SetKalmanDone(kTRUE);
  }

  return kTRUE;
}

//_________________________________________________
Float_t AliAlgRes::GetXLab(int i) const
{
  // cluster lab X
  return Abs(fX[i])*Cos(fAlpha[i]) - fY[i]*Sin(fAlpha[i]);
}

//_________________________________________________
Float_t AliAlgRes::GetYLab(int i) const
{
  // cluster lab Y
  return Abs(fX[i])*Sin(fAlpha[i]) + fY[i]*Cos(fAlpha[i]);
}

//_________________________________________________
Float_t AliAlgRes::GetZLab(int i) const
{
  // cluster lab Z
  return fZ[i];
}
