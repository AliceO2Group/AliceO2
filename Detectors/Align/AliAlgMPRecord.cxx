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

#include "AliAlgMPRecord.h"
#include "AliAlgAux.h"
#include "AliAlgTrack.h"
#include "AliLog.h"
#include <TMath.h>
#include <stdio.h>


using namespace TMath;
using namespace AliAlgAux;

//_________________________________________________________
AliAlgMPRecord::AliAlgMPRecord()
  :fTrackID(0)
  ,fTimeStamp(0)
  ,fNResid(0)
  ,fNVarLoc(0)
  ,fNVarGlo(0)
  ,fNDLocTot(0)
  ,fNDGloTot(0)
  ,fNMeas(0)
  ,fChi2Ini(0)
  ,fQ2Pt(0)
  ,fTgl(0)
  ,fNDLoc(0)
  ,fNDGlo(0)
  ,fVolID(0)
  ,fResid(0)
  ,fResErr(0)
  ,fIDLoc(0)
  ,fIDGlo(0)
  ,fDLoc(0)
  ,fDGlo(0)
   //
  ,fNResidBook(0)
  ,fNDLocTotBook(0)
  ,fNDGloTotBook(0)
{
  // def c-tor
  
}

//_________________________________________________________
AliAlgMPRecord::~AliAlgMPRecord()
{
  // d-tor
  delete[] fNDLoc;
  delete[] fNDGlo;
  delete[] fVolID;
  delete[] fResid;
  delete[] fResErr;
  delete[] fIDLoc;
  delete[] fIDGlo;
  delete[] fDLoc;
  delete[] fDGlo;
}

//_________________________________________________________
void AliAlgMPRecord::DummyRecord(Float_t res, Float_t err, Float_t dGlo, Int_t labGlo)
{
  // create dummy residuals record
  if (!fNDGlo) Resize(1,1,1);
  fChi2Ini = 0;
  fNMeas = 1;
  fNResid = 1;
  fNVarLoc = 0;
  fNVarGlo = 1;
  fIDGlo[0] = labGlo;
  fDGlo[0] = dGlo;
  fNDGlo[0] = 1;
  fVolID[0] = -1;
  fResid[0] = res;
  fResErr[0] = err;
  //
  fIDLoc[0] = 0;
  fNDLoc[0] = 0;
  fDLoc[0] = 0;
  fNDGloTot = 1;
  fNDLocTot = 0;
  //
}

//_________________________________________________________
Bool_t AliAlgMPRecord::FillTrack(const AliAlgTrack* trc, const Int_t *id2Lab)
{
  // fill track info, optionally substitutind glopar par ID by label
  //
  if (!trc->GetDerivDone()) {
    AliError("Track derivatives are not yet evaluated");
    return kFALSE;
  }
  fNVarLoc = trc->GetNLocPar();    // number of local degrees of freedom in the track
  fNResid = 0;
  fNDLocTot = 0;
  fNDGloTot = 0;
  fChi2Ini = trc->GetChi2Ini();
  fQ2Pt = trc->GetSigned1Pt();
  fTgl  = trc->GetTgl();
  fNMeas = 0;
  SetCosmic(trc->IsCosmic());
  // 1) check sizes for buffers, expand if needed
  int np = trc->GetNPoints();
  int nres  = 0;
  int nlocd = 0;
  int nglod = 0;
  for (int ip=np;ip--;) {
    AliAlgPoint* pnt = trc->GetPoint(ip);
    int ngl = pnt->GetNGloDOFs();       // number of DOF's this point depends on
    if (pnt->ContainsMeasurement()) {
      nres    += 2;                     // every point has 2 residuals
      nlocd   += fNVarLoc+fNVarLoc;     // each residual has max fNVarLoc local derivatives
      nglod   += ngl + ngl;             // number of global derivatives
      fNMeas++;
    }
    if (pnt->ContainsMaterial()) {
      int nmatpar = pnt->GetNMatPar();
      nres    += nmatpar;               // each point with materials has nmatpar fake residuals 
      nlocd   += nmatpar;               // and nmatpar non-0 local derivatives (orthogonal)
    }
  }
  //
  Resize(nres,nlocd,nglod);
  int nParETP = trc->GetNLocExtPar();   // numnber of local parameters for reference track param
  //
  const int* gloParID = trc->GetGloParID(); // IDs of global DOFs this track depends on
  for (int ip=0;ip<np;ip++) {
    AliAlgPoint* pnt = trc->GetPoint(ip);
    if (pnt->ContainsMeasurement()) {
      int gloOffs = pnt->GetDGloOffs(); // 1st entry of global derivatives for this point
      int nDGlo   = pnt->GetNGloDOFs(); // number of global derivatives (number of DOFs it depends on)
      if (!pnt->IsStatOK()) pnt->IncrementStat();
      //
      for (int idim=0;idim<2;idim++) { // 2 dimensional orthogonal measurement
	fNDGlo[fNResid] = 0;
	fVolID[fNResid] = pnt->GetSensor()->GetVolID()+1;
	// 
	// measured residual/error
	fResid[ fNResid] = trc->GetResidual(idim,ip);
	fResErr[fNResid] = Sqrt(pnt->GetErrDiag(idim));
	// 
	// derivatives over local params
	const double* deriv  = trc->GetDResDLoc(idim,ip);  // array of Dresidual/Dparams_loc
	int nnon0 = 0;
	for (int j=0;j<nParETP;j++) {       // derivatives over reference track parameters
	  if (IsZeroAbs(deriv[j])) continue;
	  nnon0++;
	  fDLoc[fNDLocTot] = deriv[j];      // store non-0 derivative
	  fIDLoc[fNDLocTot] = j;            // and variable id
	  fNDLocTot++;
	}
	int lp0 = pnt->GetMinLocVarID();    // point may depend on material variables starting from this one
	int lp1 = pnt->GetMaxLocVarID();    // and up to this one (exclusive)
	for (int j=lp0;j<lp1;j++) {         // derivatives over material variables
	  if (IsZeroAbs(deriv[j])) continue;
	  nnon0++;
	  fDLoc[fNDLocTot] = deriv[j];    // store non-0 derivative
	  fIDLoc[fNDLocTot] = j;          // and variable id
	  fNDLocTot++;
	}
	//
	fNDLoc[fNResid] = nnon0;          // local derivatives done, store their number for this residual
	//
	// derivatives over global params
	nnon0 = 0;
	deriv = trc->GetDResDGlo(idim, gloOffs);
	const int* gloIDP = gloParID + gloOffs;
	for (int j=0;j<nDGlo;j++) {
	  if (IsZeroAbs(deriv[j])) continue;
	  nnon0++;
	  fDGlo[ fNDGloTot] = deriv[j];        // value of derivative
	  fIDGlo[fNDGloTot] = id2Lab ? id2Lab[gloIDP[j]] : gloIDP[j]+1;       // global DOF ID
	  fNDGloTot++;
	}
	fNDGlo[fNResid] = nnon0;
	//
	fNResid++;
      }
    }
    if (pnt->ContainsMaterial()) {     // material point can add 4 or 5 otrhogonal pseudo-measurements
      int nmatpar = pnt->GetNMatPar();  // residuals (correction expectation value)
      //      const float* expMatCorr = pnt->GetMatCorrExp(); // expected corrections (diagonalized)
      const float* expMatCov  = pnt->GetMatCorrCov(); // their diagonalized error matrix
      int offs  = pnt->GetMaxLocVarID() - nmatpar;    // start of material variables
      // here all derivatives are 1 = dx/dx
      for (int j=0;j<nmatpar;j++) {
	fNDGlo[fNResid]  = 0;                       // mat corrections don't depend on global params
	fVolID[fNResid]  = 0;                       // not associated to global parameter
	fResid[fNResid]  = 0;                       // expectation for MS effects is 0 
	fResErr[fNResid] = Sqrt(expMatCov[j]);
	fNDLoc[fNResid] = 1;                        // only 1 non-0 derivative
	fDLoc[fNDLocTot] = 1.0;
	fIDLoc[fNDLocTot]  = offs + j;              // variable id
	fNDLocTot++;
	fNResid++;
      }
    }
  }  
  //
  if (!fNDGloTot) {
    AliInfo("Track does not depend on free global parameters, discard");
    return kFALSE;
  }
  return kTRUE;
}

//________________________________________________
void AliAlgMPRecord::Resize(Int_t nresid, Int_t nloc, Int_t nglo)
{
  // resize container
  if (nresid>fNResidBook) {
    delete[] fNDLoc;
    delete[] fNDGlo;
    delete[] fVolID;
    delete[] fResid;
    delete[] fResErr;
    fNDLoc = new Short_t[nresid];
    fNDGlo = new Int_t[nresid];
    fVolID = new Int_t[nresid];
    fResid   = new Float_t[nresid];
    fResErr  = new Float_t[nresid];
    fNResidBook = nresid;
    memset(fNDLoc,0,nresid*sizeof(Short_t));
    memset(fNDGlo,0,nresid*sizeof(Int_t));
    memset(fVolID,0,nresid*sizeof(Int_t));
    memset(fResid,0,nresid*sizeof(Float_t));
    memset(fResErr,0,nresid*sizeof(Float_t));
  }
  if (nloc>fNDLocTotBook) {
    delete[] fIDLoc;
    delete[] fDLoc;
    fIDLoc = new Short_t[nloc];
    fDLoc  = new Float_t[nloc];
    fNDLocTotBook = nloc;
    memset(fIDLoc,0,nloc*sizeof(Short_t));
    memset(fDLoc,0,nloc*sizeof(Float_t));
  }
  if (nglo>fNDGloTotBook) {
    delete[] fIDGlo;
    delete[] fDGlo;
    fIDGlo = new Int_t[nglo];
    fDGlo  = new Float_t[nglo];
    fNDGloTotBook = nglo;
    memset(fIDGlo,0,nglo*sizeof(Int_t));
    memset(fDGlo,0,nglo*sizeof(Float_t));
  }
  //
}

//____________________________________________
void AliAlgMPRecord::Clear(const Option_t *)
{
  // reset record
  TObject::Clear();
  ResetBit(0xffffffff);
  fNResid = 0;
  fNVarLoc = 0;
  fNVarGlo = 0;
  fNDLocTot = 0;
  fNDGloTot = 0;
}

//____________________________________________
void AliAlgMPRecord::Print(const Option_t *) const
{
  // print info
  //
  printf("Track %d Event TimeStamp:%d Run:%d\n",
	 fTrackID,fTimeStamp,GetRun());
  printf("Nmeas:%3d Q/pt:%+.2e Tgl:%+.2e Chi2Ini:%.1f\n",fNMeas,fQ2Pt,fTgl,fChi2Ini);
  printf("NRes: %3d NLoc: %3d NGlo:%3d | Stored: Loc:%3d Glo:%5d\n",
	 fNResid,fNVarLoc,fNVarGlo,fNDLocTot,fNDGloTot);
  //
  int curLoc=0,curGlo=0;
  const int kNColLoc=5;
  for (int ir=0;ir<fNResid;ir++) {
    int ndloc = fNDLoc[ir], ndglo = fNDGlo[ir];
    printf("Res:%3d %+e (%+e) | NDLoc:%3d NDGlo:%4d [VID:%5d]\n",
	   ir,fResid[ir],fResErr[ir],ndloc,ndglo,GetVolID(ir));
    //
    printf("Local Derivatives:\n");
    Bool_t eolOK = kTRUE;
    for (int id=0;id<ndloc;id++) {
      int jd = id+curLoc;
      printf("[%3d] %+.2e  ",fIDLoc[jd],fDLoc[jd]);
      if (((id+1)%kNColLoc)==0) {printf("\n"); eolOK = kTRUE;}
      else eolOK = kFALSE;
    }
    if (!eolOK) printf("\n");
    curLoc += ndloc;
    //
    //
    printf("Global Derivatives:\n");
    //    eolOK = kTRUE;
    //    const int kNColGlo=6;
    int prvID = -1;
    for (int id=0;id<ndglo;id++) {
      int jd = id+curGlo;
      //      eolOK==kFALSE;
      if (prvID>fIDGlo[jd]%100) {printf("\n");/* eolOK = kTRUE;*/}
      printf("[%5d] %+.2e  ",fIDGlo[jd],fDGlo[jd]);
      //      if (((id+1)%kNColGlo)==0) 
      //      else eolOK = kFALSE;
      prvID=fIDGlo[jd]%100;
    }
    //    if (!eolOK) printf("\n");
    printf("\n");
    curGlo += ndglo;
    //
  }
}
