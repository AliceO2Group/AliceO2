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

#include "AliMagWrapCheb.h"
#include "AliLog.h"
#include <TSystem.h>
#include <TArrayF.h>
#include <TArrayI.h>

ClassImp(AliMagWrapCheb)

//__________________________________________________________________________________________
AliMagWrapCheb::AliMagWrapCheb() : 
fNParamsSol(0),fNZSegSol(0),fNPSegSol(0),fNRSegSol(0),
  fSegZSol(0),fSegPSol(0),fSegRSol(0),
  fBegSegPSol(0),fNSegPSol(0),fBegSegRSol(0),fNSegRSol(0),fSegIDSol(0),fMinZSol(1.e6),fMaxZSol(-1.e6),fParamsSol(0),fMaxRSol(0),
//
  fNParamsTPC(0),fNZSegTPC(0),fNPSegTPC(0),fNRSegTPC(0),
  fSegZTPC(0),fSegPTPC(0),fSegRTPC(0),
  fBegSegPTPC(0),fNSegPTPC(0),fBegSegRTPC(0),fNSegRTPC(0),fSegIDTPC(0),fMinZTPC(1.e6),fMaxZTPC(-1.e6),fParamsTPC(0),fMaxRTPC(0),
//
  fNParamsTPCRat(0),fNZSegTPCRat(0),fNPSegTPCRat(0),fNRSegTPCRat(0),
  fSegZTPCRat(0),fSegPTPCRat(0),fSegRTPCRat(0),
  fBegSegPTPCRat(0),fNSegPTPCRat(0),fBegSegRTPCRat(0),fNSegRTPCRat(0),fSegIDTPCRat(0),fMinZTPCRat(1.e6),fMaxZTPCRat(-1.e6),fParamsTPCRat(0),fMaxRTPCRat(0),
//
  fNParamsDip(0),fNZSegDip(0),fNYSegDip(0),fNXSegDip(0),
  fSegZDip(0),fSegYDip(0),fSegXDip(0),
  fBegSegYDip(0),fNSegYDip(0),fBegSegXDip(0),fNSegXDip(0),fSegIDDip(0),fMinZDip(1.e6),fMaxZDip(-1.e6),fParamsDip(0)
//
#ifdef _MAGCHEB_CACHE_
  ,fCacheSol(0),fCacheDip(0),fCacheTPCInt(0),fCacheTPCRat(0)
#endif
//
{
  // default constructor
}

//__________________________________________________________________________________________
AliMagWrapCheb::AliMagWrapCheb(const AliMagWrapCheb& src) : 
  TNamed(src),
  fNParamsSol(0),fNZSegSol(0),fNPSegSol(0),fNRSegSol(0),
  fSegZSol(0),fSegPSol(0),fSegRSol(0),
  fBegSegPSol(0),fNSegPSol(0),fBegSegRSol(0),fNSegRSol(0),fSegIDSol(0),fMinZSol(1.e6),fMaxZSol(-1.e6),fParamsSol(0),fMaxRSol(0),
//
  fNParamsTPC(0),fNZSegTPC(0),fNPSegTPC(0),fNRSegTPC(0),
  fSegZTPC(0),fSegPTPC(0),fSegRTPC(0),
  fBegSegPTPC(0),fNSegPTPC(0),fBegSegRTPC(0),fNSegRTPC(0),fSegIDTPC(0),fMinZTPC(1.e6),fMaxZTPC(-1.e6),fParamsTPC(0),fMaxRTPC(0),
//
  fNParamsTPCRat(0),fNZSegTPCRat(0),fNPSegTPCRat(0),fNRSegTPCRat(0),
  fSegZTPCRat(0),fSegPTPCRat(0),fSegRTPCRat(0),
  fBegSegPTPCRat(0),fNSegPTPCRat(0),fBegSegRTPCRat(0),fNSegRTPCRat(0),fSegIDTPCRat(0),fMinZTPCRat(1.e6),fMaxZTPCRat(-1.e6),fParamsTPCRat(0),fMaxRTPCRat(0),
//
  fNParamsDip(0),fNZSegDip(0),fNYSegDip(0),fNXSegDip(0),
  fSegZDip(0),fSegYDip(0),fSegXDip(0),
  fBegSegYDip(0),fNSegYDip(0),fBegSegXDip(0),fNSegXDip(0),fSegIDDip(0),fMinZDip(1.e6),fMaxZDip(-1.e6),fParamsDip(0)
//
#ifdef _MAGCHEB_CACHE_
  ,fCacheSol(0),fCacheDip(0),fCacheTPCInt(0),fCacheTPCRat(0)
#endif
{
  // copy constructor
  CopyFrom(src);
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::CopyFrom(const AliMagWrapCheb& src) 
{ 
  // copy method
  Clear();
  SetName(src.GetName());
  SetTitle(src.GetTitle());
  //
  fNParamsSol    = src.fNParamsSol;
  fNZSegSol      = src.fNZSegSol;
  fNPSegSol      = src.fNPSegSol;
  fNRSegSol      = src.fNRSegSol;  
  fMinZSol       = src.fMinZSol;
  fMaxZSol       = src.fMaxZSol;
  fMaxRSol       = src.fMaxRSol;
  if (src.fNParamsSol) {
    memcpy(fSegZSol   = new Float_t[fNZSegSol], src.fSegZSol, sizeof(Float_t)*fNZSegSol);
    memcpy(fSegPSol   = new Float_t[fNPSegSol], src.fSegPSol, sizeof(Float_t)*fNPSegSol);
    memcpy(fSegRSol   = new Float_t[fNRSegSol], src.fSegRSol, sizeof(Float_t)*fNRSegSol);
    memcpy(fBegSegPSol= new Int_t[fNZSegSol], src.fBegSegPSol, sizeof(Int_t)*fNZSegSol);
    memcpy(fNSegPSol  = new Int_t[fNZSegSol], src.fNSegPSol, sizeof(Int_t)*fNZSegSol);
    memcpy(fBegSegRSol= new Int_t[fNPSegSol], src.fBegSegRSol, sizeof(Int_t)*fNPSegSol);
    memcpy(fNSegRSol  = new Int_t[fNPSegSol], src.fNSegRSol, sizeof(Int_t)*fNPSegSol);
    memcpy(fSegIDSol  = new Int_t[fNRSegSol], src.fSegIDSol, sizeof(Int_t)*fNRSegSol);
    fParamsSol        = new TObjArray(fNParamsSol);
    for (int i=0;i<fNParamsSol;i++) fParamsSol->AddAtAndExpand(new AliCheb3D(*src.GetParamSol(i)),i);
  }
  //
  fNParamsTPC    = src.fNParamsTPC;
  fNZSegTPC      = src.fNZSegTPC;
  fNPSegTPC      = src.fNPSegTPC;
  fNRSegTPC      = src.fNRSegTPC;  
  fMinZTPC       = src.fMinZTPC;
  fMaxZTPC       = src.fMaxZTPC;
  fMaxRTPC       = src.fMaxRTPC;
  if (src.fNParamsTPC) {
    memcpy(fSegZTPC   = new Float_t[fNZSegTPC], src.fSegZTPC, sizeof(Float_t)*fNZSegTPC);
    memcpy(fSegPTPC   = new Float_t[fNPSegTPC], src.fSegPTPC, sizeof(Float_t)*fNPSegTPC);
    memcpy(fSegRTPC   = new Float_t[fNRSegTPC], src.fSegRTPC, sizeof(Float_t)*fNRSegTPC);
    memcpy(fBegSegPTPC= new Int_t[fNZSegTPC], src.fBegSegPTPC, sizeof(Int_t)*fNZSegTPC);
    memcpy(fNSegPTPC  = new Int_t[fNZSegTPC], src.fNSegPTPC, sizeof(Int_t)*fNZSegTPC);
    memcpy(fBegSegRTPC= new Int_t[fNPSegTPC], src.fBegSegRTPC, sizeof(Int_t)*fNPSegTPC);
    memcpy(fNSegRTPC  = new Int_t[fNPSegTPC], src.fNSegRTPC, sizeof(Int_t)*fNPSegTPC);
    memcpy(fSegIDTPC  = new Int_t[fNRSegTPC], src.fSegIDTPC, sizeof(Int_t)*fNRSegTPC);
    fParamsTPC        = new TObjArray(fNParamsTPC);
    for (int i=0;i<fNParamsTPC;i++) fParamsTPC->AddAtAndExpand(new AliCheb3D(*src.GetParamTPCInt(i)),i);
  }
  //
  fNParamsTPCRat    = src.fNParamsTPCRat;
  fNZSegTPCRat      = src.fNZSegTPCRat;
  fNPSegTPCRat      = src.fNPSegTPCRat;
  fNRSegTPCRat      = src.fNRSegTPCRat;  
  fMinZTPCRat       = src.fMinZTPCRat;
  fMaxZTPCRat       = src.fMaxZTPCRat;
  fMaxRTPCRat       = src.fMaxRTPCRat;
  if (src.fNParamsTPCRat) {
    memcpy(fSegZTPCRat   = new Float_t[fNZSegTPCRat], src.fSegZTPCRat, sizeof(Float_t)*fNZSegTPCRat);
    memcpy(fSegPTPCRat   = new Float_t[fNPSegTPCRat], src.fSegPTPCRat, sizeof(Float_t)*fNPSegTPCRat);
    memcpy(fSegRTPCRat   = new Float_t[fNRSegTPCRat], src.fSegRTPCRat, sizeof(Float_t)*fNRSegTPCRat);
    memcpy(fBegSegPTPCRat= new Int_t[fNZSegTPCRat], src.fBegSegPTPCRat, sizeof(Int_t)*fNZSegTPCRat);
    memcpy(fNSegPTPCRat  = new Int_t[fNZSegTPCRat], src.fNSegPTPCRat, sizeof(Int_t)*fNZSegTPCRat);
    memcpy(fBegSegRTPCRat= new Int_t[fNPSegTPCRat], src.fBegSegRTPCRat, sizeof(Int_t)*fNPSegTPCRat);
    memcpy(fNSegRTPCRat  = new Int_t[fNPSegTPCRat], src.fNSegRTPCRat, sizeof(Int_t)*fNPSegTPCRat);
    memcpy(fSegIDTPCRat  = new Int_t[fNRSegTPCRat], src.fSegIDTPCRat, sizeof(Int_t)*fNRSegTPCRat);
    fParamsTPCRat        = new TObjArray(fNParamsTPCRat);
    for (int i=0;i<fNParamsTPCRat;i++) fParamsTPCRat->AddAtAndExpand(new AliCheb3D(*src.GetParamTPCRatInt(i)),i);
  }
  //
  fNParamsDip    = src.fNParamsDip;
  fNZSegDip      = src.fNZSegDip;
  fNYSegDip      = src.fNYSegDip;
  fNXSegDip      = src.fNXSegDip;  
  fMinZDip       = src.fMinZDip;
  fMaxZDip       = src.fMaxZDip;
  if (src.fNParamsDip) {
    memcpy(fSegZDip   = new Float_t[fNZSegDip], src.fSegZDip, sizeof(Float_t)*fNZSegDip);
    memcpy(fSegYDip   = new Float_t[fNYSegDip], src.fSegYDip, sizeof(Float_t)*fNYSegDip);
    memcpy(fSegXDip   = new Float_t[fNXSegDip], src.fSegXDip, sizeof(Float_t)*fNXSegDip);
    memcpy(fBegSegYDip= new Int_t[fNZSegDip], src.fBegSegYDip, sizeof(Int_t)*fNZSegDip);
    memcpy(fNSegYDip  = new Int_t[fNZSegDip], src.fNSegYDip, sizeof(Int_t)*fNZSegDip);
    memcpy(fBegSegXDip= new Int_t[fNYSegDip], src.fBegSegXDip, sizeof(Int_t)*fNYSegDip);
    memcpy(fNSegXDip  = new Int_t[fNYSegDip], src.fNSegXDip, sizeof(Int_t)*fNYSegDip);
    memcpy(fSegIDDip  = new Int_t[fNXSegDip], src.fSegIDDip, sizeof(Int_t)*fNXSegDip);
    fParamsDip        = new TObjArray(fNParamsDip);
    for (int i=0;i<fNParamsDip;i++) fParamsDip->AddAtAndExpand(new AliCheb3D(*src.GetParamDip(i)),i);
  }
  //
}

//__________________________________________________________________________________________
AliMagWrapCheb& AliMagWrapCheb::operator=(const AliMagWrapCheb& rhs)
{
  // assignment
  if (this != &rhs) {  
    Clear();
    CopyFrom(rhs);
  }
  return *this;  
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::Clear(const Option_t *)
{
  // clear all dynamic parts
  if (fNParamsSol) {
    fParamsSol->SetOwner(kTRUE);
    delete   fParamsSol;  fParamsSol = 0;
    delete[] fSegZSol;    fSegZSol   = 0;
    delete[] fSegPSol;    fSegPSol   = 0;
    delete[] fSegRSol;    fSegRSol   = 0;
    delete[] fBegSegPSol; fBegSegPSol = 0;
    delete[] fNSegPSol;   fNSegPSol   = 0;
    delete[] fBegSegRSol; fBegSegRSol = 0;
    delete[] fNSegRSol;   fNSegRSol   = 0;
    delete[] fSegIDSol;   fSegIDSol   = 0;   
  }
  fNParamsSol = fNZSegSol = fNPSegSol = fNRSegSol = 0;
  fMinZSol = 1e6;
  fMaxZSol = -1e6;
  fMaxRSol = 0;
  //
  if (fNParamsTPC) {
    fParamsTPC->SetOwner(kTRUE);
    delete   fParamsTPC;  fParamsTPC = 0;
    delete[] fSegZTPC;    fSegZTPC   = 0;
    delete[] fSegPTPC;    fSegPTPC   = 0;
    delete[] fSegRTPC;    fSegRTPC   = 0;
    delete[] fBegSegPTPC; fBegSegPTPC = 0;
    delete[] fNSegPTPC;   fNSegPTPC   = 0;
    delete[] fBegSegRTPC; fBegSegRTPC = 0;
    delete[] fNSegRTPC;   fNSegRTPC   = 0;
    delete[] fSegIDTPC;   fSegIDTPC   = 0;   
  }
  fNParamsTPC = fNZSegTPC = fNPSegTPC = fNRSegTPC = 0;
  fMinZTPC = 1e6;
  fMaxZTPC = -1e6;
  fMaxRTPC = 0;
  //
  if (fNParamsTPCRat) {
    fParamsTPCRat->SetOwner(kTRUE);
    delete   fParamsTPCRat;  fParamsTPCRat = 0;
    delete[] fSegZTPCRat;    fSegZTPCRat   = 0;
    delete[] fSegPTPCRat;    fSegPTPCRat   = 0;
    delete[] fSegRTPCRat;    fSegRTPCRat   = 0;
    delete[] fBegSegPTPCRat; fBegSegPTPCRat = 0;
    delete[] fNSegPTPCRat;   fNSegPTPCRat   = 0;
    delete[] fBegSegRTPCRat; fBegSegRTPCRat = 0;
    delete[] fNSegRTPCRat;   fNSegRTPCRat   = 0;
    delete[] fSegIDTPCRat;   fSegIDTPCRat   = 0;   
  }
  fNParamsTPCRat = fNZSegTPCRat = fNPSegTPCRat = fNRSegTPCRat = 0;
  fMinZTPCRat = 1e6;
  fMaxZTPCRat = -1e6;
  fMaxRTPCRat = 0;
  //
  if (fNParamsDip) {
    fParamsDip->SetOwner(kTRUE);
    delete   fParamsDip;  fParamsDip = 0;
    delete[] fSegZDip;    fSegZDip   = 0;
    delete[] fSegYDip;    fSegYDip   = 0; 
    delete[] fSegXDip;    fSegXDip   = 0;
    delete[] fBegSegYDip; fBegSegYDip = 0;
    delete[] fNSegYDip;   fNSegYDip   = 0;
    delete[] fBegSegXDip; fBegSegXDip = 0; 
    delete[] fNSegXDip;   fNSegXDip   = 0;
    delete[] fSegIDDip;   fSegIDDip   = 0;
  }
  fNParamsDip = fNZSegDip = fNYSegDip = fNXSegDip = 0;
  fMinZDip = 1e6;
  fMaxZDip = -1e6;
  //
#ifdef _MAGCHEB_CACHE_
  fCacheSol = 0;
  fCacheDip = 0;
  fCacheTPCInt = 0;
  fCacheTPCRat = 0;
#endif
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::Field(const Double_t *xyz, Double_t *b) const
{
  // compute field in cartesian coordinates. If point is outside of the parameterized region
  // get it at closest valid point
  Double_t rphiz[3];
  //
#ifndef _BRING_TO_BOUNDARY_  // exact matching to fitted volume is requested
  b[0] = b[1] = b[2] = 0;
#endif 
  //
  if (xyz[2]>fMinZSol) {
    CartToCyl(xyz,rphiz);
    //
#ifdef _MAGCHEB_CACHE_
    if (fCacheSol && fCacheSol->IsInside(rphiz)) 
      fCacheSol->Eval(rphiz,b);
    else
#endif //_MAGCHEB_CACHE_
      FieldCylSol(rphiz,b);
    // convert field to cartesian system
    CylToCartCylB(rphiz, b,b);
    return;
  }
  //
#ifdef _MAGCHEB_CACHE_
  if (fCacheDip && fCacheDip->IsInside(xyz)) {
    fCacheDip->Eval(xyz,b); // check the cache first
    return;
  }
#else //_MAGCHEB_CACHE_
  AliCheb3D* fCacheDip = 0;
#endif //_MAGCHEB_CACHE_
  int iddip = FindDipSegment(xyz);
  if (iddip>=0) {
    fCacheDip = GetParamDip(iddip);
    //
#ifndef _BRING_TO_BOUNDARY_
    if (!fCacheDip->IsInside(xyz)) return;
#endif //_BRING_TO_BOUNDARY_
    //
    fCacheDip->Eval(xyz,b); 
  }
  //
}

//__________________________________________________________________________________________
Double_t AliMagWrapCheb::GetBz(const Double_t *xyz) const
{
  // compute Bz for the point in cartesian coordinates. If point is outside of the parameterized region
  // get it at closest valid point
  Double_t rphiz[3];
  //
  if (xyz[2]>fMinZSol) {
    //
    CartToCyl(xyz,rphiz);
    //
#ifdef _MAGCHEB_CACHE_
    if (fCacheSol && fCacheSol->IsInside(rphiz)) return fCacheSol->Eval(rphiz,2);
#endif //_MAGCHEB_CACHE_
    return FieldCylSolBz(rphiz);
  }
  //
#ifdef _MAGCHEB_CACHE_
  if (fCacheDip && fCacheDip->IsInside(xyz)) return fCacheDip->Eval(xyz,2); // check the cache first
  //
#else //_MAGCHEB_CACHE_
  AliCheb3D* fCacheDip = 0;
#endif //_MAGCHEB_CACHE_
  //
  int iddip = FindDipSegment(xyz);
  if (iddip>=0) {
    fCacheDip = GetParamDip(iddip);
    //
#ifndef _BRING_TO_BOUNDARY_
    if (!fCacheDip->IsInside(xyz)) return 0.;
#endif // _BRING_TO_BOUNDARY_
    //
    return fCacheDip->Eval(xyz,2);
  //
  }
  //
  return 0;
  //
}


//__________________________________________________________________________________________
void AliMagWrapCheb::Print(Option_t *) const
{
  // print info
  printf("Alice magnetic field parameterized by Chebyshev polynomials\n");
  printf("Segmentation for Solenoid (%+.2f<Z<%+.2f cm | R<%.2f cm)\n",fMinZSol,fMaxZSol,fMaxRSol);
  //
  if (fParamsSol) {
    for (int i=0;i<fNParamsSol;i++) {
      printf("SOL%4d ",i);
      GetParamSol(i)->Print();
    }
  }
  //
  printf("Segmentation for TPC field integral (%+.2f<Z<%+.2f cm | R<%.2f cm)\n",fMinZTPC,fMaxZTPC,fMaxRTPC);
  //
  if (fParamsTPC) {
    for (int i=0;i<fNParamsTPC;i++) {
      printf("TPC%4d ",i);
      GetParamTPCInt(i)->Print();
    }
  }
  //
  printf("Segmentation for TPC field ratios integral (%+.2f<Z<%+.2f cm | R<%.2f cm)\n",fMinZTPCRat,fMaxZTPCRat,fMaxRTPCRat);
  //
  if (fParamsTPCRat) {
    for (int i=0;i<fNParamsTPCRat;i++) {
      printf("TPCRat%4d ",i);
      GetParamTPCRatInt(i)->Print();
    }
  }
  //
  printf("Segmentation for Dipole (%+.2f<Z<%+.2f cm)\n",fMinZDip,fMaxZDip);
  if (fParamsDip) {
    for (int i=0;i<fNParamsDip;i++) {
      printf("DIP%4d ",i);
      GetParamDip(i)->Print();
    }
  }
  //
}

//__________________________________________________________________________________________________
Int_t AliMagWrapCheb::FindDipSegment(const Double_t *xyz) const 
{
  // find the segment containing point xyz. If it is outside find the closest segment 
  if (!fNParamsDip) return -1;
  int xid,yid,zid = TMath::BinarySearch(fNZSegDip,fSegZDip,(Float_t)xyz[2]); // find zsegment
  //
  Bool_t reCheck = kFALSE;
  while(1) {
    int ysegBeg = fBegSegYDip[zid];
    //
    for (yid=0;yid<fNSegYDip[zid];yid++) if (xyz[1]<fSegYDip[ysegBeg+yid]) break;
    if ( --yid < 0 ) yid = 0;
    yid +=  ysegBeg;
    //
    int xsegBeg = fBegSegXDip[yid];
    for (xid=0;xid<fNSegXDip[yid];xid++) if (xyz[0]<fSegXDip[xsegBeg+xid]) break;
    //
    if ( --xid < 0) xid = 0;
    xid +=  xsegBeg;
    //
    // to make sure that due to the precision problems we did not pick the next Zbin    
    if (!reCheck && (xyz[2] - fSegZDip[zid] < 3.e-5) && zid &&
	!GetParamDip(fSegIDDip[xid])->IsInside(xyz)) {  // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    } 
    break;
  }
  //  AliInfo(Form("%+.2f %+.2f %+.2f %d %d %d %4d",xyz[0],xyz[1],xyz[2],xid,yid,zid,fSegIDDip[xid]));
  return fSegIDDip[xid];
}

//__________________________________________________________________________________________________
Int_t AliMagWrapCheb::GetDipSegmentsForZSlice(int zid, TObjArray& arr) const
{
  if (zid<0 || zid>=fNZSegDip) return -1;
  arr.Clear();
  arr.SetOwner(kFALSE);
  int ysegBeg = fBegSegYDip[zid];
  for (int yid=0;yid<fNSegYDip[zid];yid++) {
    int yid1 = yid + ysegBeg;
    int xsegBeg = fBegSegXDip[yid1];
    for (int xid=0;xid<fNSegXDip[yid1];xid++) {
      int xid1 = xid + xsegBeg;
      //      printf("#%2d Adding segment %d\n",arr.GetEntriesFast(),xid1);
      //      GetParamDip(fSegIDDip[xid1])->Print();
      arr.AddLast( GetParamDip(fSegIDDip[xid1]) );
    }    
  }
  return arr.GetEntriesFast();
}



//__________________________________________________________________________________________________
Int_t AliMagWrapCheb::FindSolSegment(const Double_t *rpz) const 
{
  // find the segment containing point xyz. If it is outside find the closest segment 
  if (!fNParamsSol) return -1;
  int rid,pid,zid = TMath::BinarySearch(fNZSegSol,fSegZSol,(Float_t)rpz[2]); // find zsegment
  //
  Bool_t reCheck = kFALSE;
  while(1) {
    int psegBeg = fBegSegPSol[zid];
    for (pid=0;pid<fNSegPSol[zid];pid++) if (rpz[1]<fSegPSol[psegBeg+pid]) break;
    if ( --pid < 0 ) pid = 0;
    pid +=  psegBeg;
    //
    int rsegBeg = fBegSegRSol[pid];
    for (rid=0;rid<fNSegRSol[pid];rid++) if (rpz[0]<fSegRSol[rsegBeg+rid]) break;
    if ( --rid < 0) rid = 0;
    rid +=  rsegBeg;
    //
    // to make sure that due to the precision problems we did not pick the next Zbin    
    if (!reCheck && (rpz[2] - fSegZSol[zid] < 3.e-5) && zid &&
	!GetParamSol(fSegIDSol[rid])->IsInside(rpz)) {  // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    } 
    break;
  }
  //  AliInfo(Form("%+.2f %+.4f %+.2f %d %d %d %4d",rpz[0],rpz[1],rpz[2],rid,pid,zid,fSegIDSol[rid]));
  return fSegIDSol[rid];
}

//__________________________________________________________________________________________________
Int_t AliMagWrapCheb::FindTPCSegment(const Double_t *rpz) const 
{
  // find the segment containing point xyz. If it is outside find the closest segment 
  if (!fNParamsTPC) return -1;
  int rid,pid,zid = TMath::BinarySearch(fNZSegTPC,fSegZTPC,(Float_t)rpz[2]); // find zsegment
  //
  Bool_t reCheck = kFALSE;
  while(1) {
    int psegBeg = fBegSegPTPC[zid];
    //
    for (pid=0;pid<fNSegPTPC[zid];pid++) if (rpz[1]<fSegPTPC[psegBeg+pid]) break;
    if ( --pid < 0 ) pid = 0;
    pid +=  psegBeg;
    //
    int rsegBeg = fBegSegRTPC[pid];
    for (rid=0;rid<fNSegRTPC[pid];rid++) if (rpz[0]<fSegRTPC[rsegBeg+rid]) break;
    if ( --rid < 0) rid = 0;
    rid +=  rsegBeg;
    //
    // to make sure that due to the precision problems we did not pick the next Zbin    
    if (!reCheck && (rpz[2] - fSegZTPC[zid] < 3.e-5) && zid &&
	!GetParamTPCInt(fSegIDTPC[rid])->IsInside(rpz)) {  // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    } 
    break;
  }
  //  AliInfo(Form("%+.2f %+.4f %+.2f %d %d %d %4d",rpz[0],rpz[1],rpz[2],rid,pid,zid,fSegIDTPC[rid]));
  return fSegIDTPC[rid];
}

//__________________________________________________________________________________________________
Int_t AliMagWrapCheb::FindTPCRatSegment(const Double_t *rpz) const 
{
  // find the segment containing point xyz. If it is outside find the closest segment 
  if (!fNParamsTPCRat) return -1;
  int rid,pid,zid = TMath::BinarySearch(fNZSegTPCRat,fSegZTPCRat,(Float_t)rpz[2]); // find zsegment
  //
  Bool_t reCheck = kFALSE;
  while(1) {
    int psegBeg = fBegSegPTPCRat[zid];
    //
    for (pid=0;pid<fNSegPTPCRat[zid];pid++) if (rpz[1]<fSegPTPCRat[psegBeg+pid]) break;
    if ( --pid < 0 ) pid = 0;
    pid +=  psegBeg;
    //
    int rsegBeg = fBegSegRTPCRat[pid];
    for (rid=0;rid<fNSegRTPCRat[pid];rid++) if (rpz[0]<fSegRTPCRat[rsegBeg+rid]) break;
    if ( --rid < 0) rid = 0;
    rid +=  rsegBeg;
    //
    // to make sure that due to the precision problems we did not pick the next Zbin    
    if (!reCheck && (rpz[2] - fSegZTPCRat[zid] < 3.e-5) && zid &&
	!GetParamTPCRatInt(fSegIDTPCRat[rid])->IsInside(rpz)) {  // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    } 
    break;
  }
  //  AliInfo(Form("%+.2f %+.4f %+.2f %d %d %d %4d",rpz[0],rpz[1],rpz[2],rid,pid,zid,fSegIDTPCRat[rid]));
  return fSegIDTPCRat[rid];
}


//__________________________________________________________________________________________
void AliMagWrapCheb::GetTPCInt(const Double_t *xyz, Double_t *b) const
{
  // compute TPC region field integral in cartesian coordinates.
  // If point is outside of the parameterized region get it at closeset valid point
  static Double_t rphiz[3];
  //
  // TPCInt region
  // convert coordinates to cyl system
  CartToCyl(xyz,rphiz);
#ifndef _BRING_TO_BOUNDARY_
  if ( (rphiz[2]>GetMaxZTPCInt()||rphiz[2]<GetMinZTPCInt()) ||
       rphiz[0]>GetMaxRTPCInt()) {for (int i=3;i--;) b[i]=0; return;}
#endif
  //
  GetTPCIntCyl(rphiz,b);
  //
  // convert field to cartesian system
  CylToCartCylB(rphiz, b,b);
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::GetTPCRatInt(const Double_t *xyz, Double_t *b) const
{
  // compute TPCRat region field integral in cartesian coordinates.
  // If point is outside of the parameterized region get it at closeset valid point
  static Double_t rphiz[3];
  //
  // TPCRatInt region
  // convert coordinates to cyl system
  CartToCyl(xyz,rphiz);
#ifndef _BRING_TO_BOUNDARY_
  if ( (rphiz[2]>GetMaxZTPCRatInt()||rphiz[2]<GetMinZTPCRatInt()) ||
       rphiz[0]>GetMaxRTPCRatInt()) {for (int i=3;i--;) b[i]=0; return;}
#endif
  //
  GetTPCRatIntCyl(rphiz,b);
  //
  // convert field to cartesian system
  CylToCartCylB(rphiz, b,b);
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::FieldCylSol(const Double_t *rphiz, Double_t *b) const
{
  // compute Solenoid field in Cylindircal coordinates
  // note: if the point is outside the volume get the field in closest parameterized point
  int id = FindSolSegment(rphiz);
  if (id>=0) {
#ifndef _MAGCHEB_CACHE_
    AliCheb3D* fCacheSol = 0;
#endif
    fCacheSol = GetParamSol(id);
    //
#ifndef _BRING_TO_BOUNDARY_  // exact matching to fitted volume is requested  
    if (!fCacheSol->IsInside(rphiz)) return;
#endif
    fCacheSol->Eval(rphiz,b);
  }
  //
}

//__________________________________________________________________________________________
Double_t AliMagWrapCheb::FieldCylSolBz(const Double_t *rphiz) const
{
  // compute Solenoid field in Cylindircal coordinates
  // note: if the point is outside the volume get the field in closest parameterized point
  int id = FindSolSegment(rphiz);
  if (id<0) return 0.;
  //
#ifndef _MAGCHEB_CACHE_
  AliCheb3D* fCacheSol = 0;
#endif
  //
  fCacheSol = GetParamSol(id);
#ifndef _BRING_TO_BOUNDARY_  
  return fCacheSol->IsInside(rphiz) ? fCacheSol->Eval(rphiz,2) : 0;
#else
  return fCacheSol->Eval(rphiz,2);
#endif
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::GetTPCIntCyl(const Double_t *rphiz, Double_t *b) const
{
  // compute field integral in TPC region in Cylindircal coordinates
  // note: the check for the point being inside the parameterized region is done outside
  //
#ifdef _MAGCHEB_CACHE_
  //  
  if (fCacheTPCInt && fCacheTPCInt->IsInside(rphiz)) {
    fCacheTPCInt->Eval(rphiz,b);
    return;
  }
#else //_MAGCHEB_CACHE_
  AliCheb3D* fCacheTPCInt = 0; 
#endif //_MAGCHEB_CACHE_
  //
  int id = FindTPCSegment(rphiz);
  if (id>=0) {
    //    if (id>=fNParamsTPC) {
    //      b[0] = b[1] = b[2] = 0;
    //      AliError(Form("Wrong TPCParam segment %d",id));
    //      b[0] = b[1] = b[2] = 0;
    //      return;
    //    }
    fCacheTPCInt = GetParamTPCInt(id);
    if (fCacheTPCInt->IsInside(rphiz)) {
      fCacheTPCInt->Eval(rphiz,b); 
      return;
    }
  }
  //
  b[0] = b[1] = b[2] = 0;
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::GetTPCRatIntCyl(const Double_t *rphiz, Double_t *b) const
{
  // compute field integral in TPCRat region in Cylindircal coordinates
  // note: the check for the point being inside the parameterized region is done outside
  //
#ifdef _MAGCHEB_CACHE_
  if (fCacheTPCRat && fCacheTPCRat->IsInside(rphiz)) {
    fCacheTPCRat->Eval(rphiz,b);
    return;
  }
#else 
  AliCheb3D* fCacheTPCRat = 0;
#endif //_MAGCHEB_CACHE_
  //
  int id = FindTPCRatSegment(rphiz);
  if (id>=0) {
    //    if (id>=fNParamsTPCRat) {
    //      AliError(Form("Wrong TPCRatParam segment %d",id));
    //      b[0] = b[1] = b[2] = 0;
    //      return;
    //    }
    fCacheTPCRat = GetParamTPCRatInt(id);
    if (fCacheTPCRat->IsInside(rphiz)) {
      fCacheTPCRat->Eval(rphiz,b); 
      return;
    }
  }
  //
  b[0] = b[1] = b[2] = 0;
  //
}


#ifdef  _INC_CREATION_ALICHEB3D_
//_______________________________________________
void AliMagWrapCheb::LoadData(const char* inpfile)
{
  // read coefficients data from the text file
  //
  TString strf = inpfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf,"r");
  if (!stream) {
    printf("Did not find input file %s\n",strf.Data());
    return;
  }
  //
  TString buffs;
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("START")) AliFatalF("Expected: \"START <name>\", found \"%s\"",buffs.Data());
  if (buffs.First(' ')>0) SetName(buffs.Data()+buffs.First(' ')+1);
  //
  // Solenoid part    -----------------------------------------------------------
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("START SOLENOID")) AliFatalF("Expected: \"START SOLENOID\", found \"%s\"",buffs.Data());
  AliCheb3DCalc::ReadLine(buffs,stream); // nparam
  int nparSol = buffs.Atoi(); 
  //
  for (int ip=0;ip<nparSol;ip++) {
    AliCheb3D* cheb = new AliCheb3D();
    cheb->LoadData(stream);
    AddParamSol(cheb);
  }
  //
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("END SOLENOID")) AliFatalF("Expected \"END SOLENOID\", found \"%s\"",buffs.Data());
  //
  // TPCInt part     -----------------------------------------------------------
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("START TPCINT")) AliFatalF("Expected: \"START TPCINT\", found \"%s\"",buffs.Data());
  //
  AliCheb3DCalc::ReadLine(buffs,stream); // nparam
  int nparTPCInt = buffs.Atoi(); 
  //
  for (int ip=0;ip<nparTPCInt;ip++) {
    AliCheb3D* cheb = new AliCheb3D();
    cheb->LoadData(stream);
    AddParamTPCInt(cheb);
  }
  //
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("END TPCINT")) AliFatalF("Expected \"END TPCINT\", found \"%s\"",buffs.Data());
  //
  // TPCRatInt part     -----------------------------------------------------------
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("START TPCRatINT")) AliFatalF("Expected: \"START TPCRatINT\", found \"%s\"",buffs.Data());
  AliCheb3DCalc::ReadLine(buffs,stream); // nparam
  int nparTPCRatInt = buffs.Atoi(); 
  //
  for (int ip=0;ip<nparTPCRatInt;ip++) {
    AliCheb3D* cheb = new AliCheb3D();
    cheb->LoadData(stream);
    AddParamTPCRatInt(cheb);
  }
  //
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("END TPCRatINT")) AliFatalF("Expected \"END TPCRatINT\", found \"%s\"",buffs.Data());
  //
  // Dipole part    -----------------------------------------------------------
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("START DIPOLE")) AliFatalF("Expected: \"START DIPOLE\", found \"%s\"",buffs.Data());
  //
  AliCheb3DCalc::ReadLine(buffs,stream); // nparam
  int nparDip = buffs.Atoi();  
  //
  for (int ip=0;ip<nparDip;ip++) {
    AliCheb3D* cheb = new AliCheb3D();
    cheb->LoadData(stream);
    AddParamDip(cheb);
  }
  //
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("END DIPOLE")) AliFatalF("Expected \"END DIPOLE\", found \"%s\"",buffs.Data());
  //
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("END ") && !buffs.Contains(GetName())) AliFatalF("Expected: \"END %s\", found \"%s\"",GetName(),buffs.Data());
  //
  // ---------------------------------------------------------------------------
  fclose(stream);
  BuildTableSol();
  BuildTableDip();
  BuildTableTPCInt();
  BuildTableTPCRatInt();
  //
  printf("Loaded magnetic field \"%s\" from %s\n",GetName(),strf.Data());
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::BuildTableSol()
{
  // build lookup table
  BuildTable(fNParamsSol,fParamsSol,
	     fNZSegSol,fNPSegSol,fNRSegSol,
	     fMinZSol,fMaxZSol, 
	     &fSegZSol,&fSegPSol,&fSegRSol,
	     &fBegSegPSol,&fNSegPSol,
	     &fBegSegRSol,&fNSegRSol, 
	     &fSegIDSol);
}

//__________________________________________________________________________________________
void AliMagWrapCheb::BuildTableDip()
{
  // build lookup table
  BuildTable(fNParamsDip,fParamsDip,
	     fNZSegDip,fNYSegDip,fNXSegDip,
	     fMinZDip,fMaxZDip, 
	     &fSegZDip,&fSegYDip,&fSegXDip,
	     &fBegSegYDip,&fNSegYDip,
	     &fBegSegXDip,&fNSegXDip, 
	     &fSegIDDip);
}

//__________________________________________________________________________________________
void AliMagWrapCheb::BuildTableTPCInt()
{
  // build lookup table
  BuildTable(fNParamsTPC,fParamsTPC,
	     fNZSegTPC,fNPSegTPC,fNRSegTPC,
	     fMinZTPC,fMaxZTPC, 
	     &fSegZTPC,&fSegPTPC,&fSegRTPC,
	     &fBegSegPTPC,&fNSegPTPC,
	     &fBegSegRTPC,&fNSegRTPC, 
	     &fSegIDTPC);
}

//__________________________________________________________________________________________
void AliMagWrapCheb::BuildTableTPCRatInt()
{
  // build lookup table
  BuildTable(fNParamsTPCRat,fParamsTPCRat,
	     fNZSegTPCRat,fNPSegTPCRat,fNRSegTPCRat,
	     fMinZTPCRat,fMaxZTPCRat, 
	     &fSegZTPCRat,&fSegPTPCRat,&fSegRTPCRat,
	     &fBegSegPTPCRat,&fNSegPTPCRat,
	     &fBegSegRTPCRat,&fNSegRTPCRat, 
	     &fSegIDTPCRat);
}

#endif

//_______________________________________________
#ifdef  _INC_CREATION_ALICHEB3D_

//__________________________________________________________________________________________
AliMagWrapCheb::AliMagWrapCheb(const char* inputFile) : 
  fNParamsSol(0),fNZSegSol(0),fNPSegSol(0),fNRSegSol(0),
  fSegZSol(0),fSegPSol(0),fSegRSol(0),
  fBegSegPSol(0),fNSegPSol(0),fBegSegRSol(0),fNSegRSol(0),fSegIDSol(0),fMinZSol(1.e6),fMaxZSol(-1.e6),fParamsSol(0),fMaxRSol(0),
//
  fNParamsTPC(0),fNZSegTPC(0),fNPSegTPC(0),fNRSegTPC(0),
  fSegZTPC(0),fSegPTPC(0),fSegRTPC(0),
  fBegSegPTPC(0),fNSegPTPC(0),fBegSegRTPC(0),fNSegRTPC(0),fSegIDTPC(0),fMinZTPC(1.e6),fMaxZTPC(-1.e6),fParamsTPC(0),fMaxRTPC(0),
//
  fNParamsTPCRat(0),fNZSegTPCRat(0),fNPSegTPCRat(0),fNRSegTPCRat(0),
  fSegZTPCRat(0),fSegPTPCRat(0),fSegRTPCRat(0),
  fBegSegPTPCRat(0),fNSegPTPCRat(0),fBegSegRTPCRat(0),fNSegRTPCRat(0),fSegIDTPCRat(0),fMinZTPCRat(1.e6),fMaxZTPCRat(-1.e6),fParamsTPCRat(0),fMaxRTPCRat(0),
//
  fNParamsDip(0),fNZSegDip(0),fNYSegDip(0),fNXSegDip(0),
  fSegZDip(0),fSegYDip(0),fSegXDip(0),
  fBegSegYDip(0),fNSegYDip(0),fBegSegXDip(0),fNSegXDip(0),fSegIDDip(0),fMinZDip(1.e6),fMaxZDip(-1.e6),fParamsDip(0)
#ifdef _MAGCHEB_CACHE_
  ,fCacheSol(0),fCacheDip(0),fCacheTPCInt(0),fCacheTPCRat(0)
#endif
//
{
  // construct from coeffs from the text file
  LoadData(inputFile);
}

//__________________________________________________________________________________________
void AliMagWrapCheb::AddParamSol(const AliCheb3D* param)
{
  // adds new parameterization piece for Sol
  // NOTE: pieces must be added strictly in increasing R then increasing Z order
  //
  if (!fParamsSol) fParamsSol = new TObjArray();
  fParamsSol->Add( (AliCheb3D*)param );
  fNParamsSol++;
  if (fMaxRSol<param->GetBoundMax(0)) fMaxRSol = param->GetBoundMax(0);
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::AddParamTPCInt(const AliCheb3D* param)
{
  // adds new parameterization piece for TPCInt
  // NOTE: pieces must be added strictly in increasing R then increasing Z order
  //
  if (!fParamsTPC) fParamsTPC = new TObjArray();
  fParamsTPC->Add( (AliCheb3D*)param);
  fNParamsTPC++;
  if (fMaxRTPC<param->GetBoundMax(0)) fMaxRTPC = param->GetBoundMax(0);
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::AddParamTPCRatInt(const AliCheb3D* param)
{
  // adds new parameterization piece for TPCRatInt
  // NOTE: pieces must be added strictly in increasing R then increasing Z order
  //
  if (!fParamsTPCRat) fParamsTPCRat = new TObjArray();
  fParamsTPCRat->Add( (AliCheb3D*)param);
  fNParamsTPCRat++;
  if (fMaxRTPCRat<param->GetBoundMax(0)) fMaxRTPCRat = param->GetBoundMax(0);
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::AddParamDip(const AliCheb3D* param)
{
  // adds new parameterization piece for Dipole
  //
  if (!fParamsDip) fParamsDip = new TObjArray();
  fParamsDip->Add( (AliCheb3D*)param);
  fNParamsDip++;
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::ResetDip()
{
  // clean Dip field (used for update)
  if (fNParamsDip) {
    delete   fParamsDip;  fParamsDip = 0;
    delete[] fSegZDip;    fSegZDip   = 0;
    delete[] fSegXDip;    fSegXDip   = 0;
    delete[] fSegYDip;    fSegYDip   = 0;
    delete[] fBegSegYDip; fBegSegYDip = 0;
    delete[] fNSegYDip;   fNSegYDip   = 0;
    delete[] fBegSegXDip; fBegSegXDip = 0;
    delete[] fNSegXDip;   fNSegXDip   = 0;
    delete[] fSegIDDip;   fSegIDDip   = 0;   
  }
  fNParamsDip = fNZSegDip = fNXSegDip = fNYSegDip = 0;
  fMinZDip = 1e6;
  fMaxZDip = -1e6;
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::ResetSol()
{
  // clean Sol field (used for update)
  if (fNParamsSol) {
    delete   fParamsSol;  fParamsSol = 0;
    delete[] fSegZSol;    fSegZSol   = 0;
    delete[] fSegPSol;    fSegPSol   = 0;
    delete[] fSegRSol;    fSegRSol   = 0;
    delete[] fBegSegPSol; fBegSegPSol = 0;
    delete[] fNSegPSol;   fNSegPSol   = 0;
    delete[] fBegSegRSol; fBegSegRSol = 0;
    delete[] fNSegRSol;   fNSegRSol   = 0;
    delete[] fSegIDSol;   fSegIDSol   = 0;   
  }
  fNParamsSol = fNZSegSol = fNPSegSol = fNRSegSol = 0;
  fMinZSol = 1e6;
  fMaxZSol = -1e6;
  fMaxRSol = 0;
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::ResetTPCInt()
{
  // clean TPC field integral (used for update)
  if (fNParamsTPC) {
    delete   fParamsTPC;  fParamsTPC = 0;
    delete[] fSegZTPC;    fSegZTPC   = 0;
    delete[] fSegPTPC;    fSegPTPC   = 0;
    delete[] fSegRTPC;    fSegRTPC   = 0;
    delete[] fBegSegPTPC; fBegSegPTPC = 0;
    delete[] fNSegPTPC;   fNSegPTPC   = 0;
    delete[] fBegSegRTPC; fBegSegRTPC = 0;
    delete[] fNSegRTPC;   fNSegRTPC   = 0;
    delete[] fSegIDTPC;   fSegIDTPC   = 0;   
  }
  fNParamsTPC = fNZSegTPC = fNPSegTPC = fNRSegTPC = 0;
  fMinZTPC = 1e6;
  fMaxZTPC = -1e6;
  fMaxRTPC = 0;
  //
}

//__________________________________________________________________________________________
void AliMagWrapCheb::ResetTPCRatInt()
{
  // clean TPCRat field integral (used for update)
  if (fNParamsTPCRat) {
    delete   fParamsTPCRat;  fParamsTPCRat = 0;
    delete[] fSegZTPCRat;    fSegZTPCRat   = 0;
    delete[] fSegPTPCRat;    fSegPTPCRat   = 0;
    delete[] fSegRTPCRat;    fSegRTPCRat   = 0;
    delete[] fBegSegPTPCRat; fBegSegPTPCRat = 0;
    delete[] fNSegPTPCRat;   fNSegPTPCRat   = 0;
    delete[] fBegSegRTPCRat; fBegSegRTPCRat = 0;
    delete[] fNSegRTPCRat;   fNSegRTPCRat   = 0;
    delete[] fSegIDTPCRat;   fSegIDTPCRat   = 0;   
  }
  fNParamsTPCRat = fNZSegTPCRat = fNPSegTPCRat = fNRSegTPCRat = 0;
  fMinZTPCRat = 1e6;
  fMaxZTPCRat = -1e6;
  fMaxRTPCRat = 0;
  //
}


//__________________________________________________
void AliMagWrapCheb::BuildTable(Int_t npar,TObjArray *parArr, Int_t &nZSeg, Int_t &nYSeg, Int_t &nXSeg,
				Float_t &minZ,Float_t &maxZ,
				Float_t **segZ,Float_t **segY,Float_t **segX,
				Int_t **begSegY,Int_t **nSegY,
				Int_t **begSegX,Int_t **nSegX,
				Int_t **segID)
{
  // build lookup table for dipole
  //
  if (npar<1) return;
  TArrayF segYArr,segXArr;
  TArrayI begSegYDipArr,begSegXDipArr;
  TArrayI nSegYDipArr,nSegXDipArr;
  TArrayI segIDArr;
  float *tmpSegZ,*tmpSegY,*tmpSegX;
  //
  // create segmentation in Z
  nZSeg = SegmentDimension(&tmpSegZ, parArr, npar, 2, 1,-1, 1,-1, 1,-1) - 1;
  nYSeg = 0;
  nXSeg = 0;
  //
  // for each Z slice create segmentation in Y
  begSegYDipArr.Set(nZSeg);
  nSegYDipArr.Set(nZSeg);
  float xyz[3];
  for (int iz=0;iz<nZSeg;iz++) {
    printf("\nZSegment#%d  %+e : %+e\n",iz,tmpSegZ[iz],tmpSegZ[iz+1]);
    int ny = SegmentDimension(&tmpSegY, parArr, npar, 1, 
			      1,-1, 1,-1, tmpSegZ[iz],tmpSegZ[iz+1]) - 1;
    segYArr.Set(ny + nYSeg);
    for (int iy=0;iy<ny;iy++) segYArr[nYSeg+iy] = tmpSegY[iy];
    begSegYDipArr[iz] = nYSeg;
    nSegYDipArr[iz] = ny;
    printf(" Found %d YSegments, to start from %d\n",ny, begSegYDipArr[iz]);
    //
    // for each slice in Z and Y create segmentation in X
    begSegXDipArr.Set(nYSeg+ny);
    nSegXDipArr.Set(nYSeg+ny);
    xyz[2] = (tmpSegZ[iz]+tmpSegZ[iz+1])/2.; // mean Z of this segment
    //
    for (int iy=0;iy<ny;iy++) {
      int isg = nYSeg+iy;
      printf("\n   YSegment#%d  %+e : %+e\n",iy, tmpSegY[iy],tmpSegY[iy+1]);
      int nx = SegmentDimension(&tmpSegX, parArr, npar, 0, 
				1,-1, tmpSegY[iy],tmpSegY[iy+1], tmpSegZ[iz],tmpSegZ[iz+1]) - 1;
      //
      segXArr.Set(nx + nXSeg);
      for (int ix=0;ix<nx;ix++) segXArr[nXSeg+ix] = tmpSegX[ix];
      begSegXDipArr[isg] = nXSeg;
      nSegXDipArr[isg] = nx;
      printf("   Found %d XSegments, to start from %d\n",nx, begSegXDipArr[isg]);
      //
      segIDArr.Set(nXSeg+nx);
      //
      // find corresponding params
      xyz[1] = (tmpSegY[iy]+tmpSegY[iy+1])/2.; // mean Y of this segment
      //
      for (int ix=0;ix<nx;ix++) {
	xyz[0] = (tmpSegX[ix]+tmpSegX[ix+1])/2.; // mean X of this segment
	for (int ipar=0;ipar<npar;ipar++) {
	  AliCheb3D* cheb = (AliCheb3D*) parArr->At(ipar);
	  if (!cheb->IsInside(xyz)) continue;
	  segIDArr[nXSeg+ix] = ipar;
	  break;
	}
      }
      nXSeg += nx;
      //
      delete[] tmpSegX;
    }
    delete[] tmpSegY;
    nYSeg += ny;
  }
  //
  minZ = tmpSegZ[0];
  maxZ = tmpSegZ[nZSeg];
  (*segZ)  = new Float_t[nZSeg];
  for (int i=nZSeg;i--;) (*segZ)[i] = tmpSegZ[i];
  delete[] tmpSegZ;
  //
  (*segY)    = new Float_t[nYSeg];
  (*segX)    = new Float_t[nXSeg];
  (*begSegY) = new Int_t[nZSeg];
  (*nSegY)   = new Int_t[nZSeg];
  (*begSegX) = new Int_t[nYSeg];
  (*nSegX)   = new Int_t[nYSeg];
  (*segID)   = new Int_t[nXSeg];
  //
  for (int i=nYSeg;i--;) (*segY)[i] = segYArr[i];
  for (int i=nXSeg;i--;) (*segX)[i] = segXArr[i];
  for (int i=nZSeg;i--;) {(*begSegY)[i] = begSegYDipArr[i]; (*nSegY)[i] = nSegYDipArr[i];}
  for (int i=nYSeg;i--;) {(*begSegX)[i] = begSegXDipArr[i]; (*nSegX)[i] = nSegXDipArr[i];}
  for (int i=nXSeg;i--;) {(*segID)[i]   = segIDArr[i];}
  //
}

/*
//__________________________________________________
void AliMagWrapCheb::BuildTableDip()
{
  // build lookup table for dipole
  //
  if (fNParamsDip<1) return;
  TArrayF segY,segX;
  TArrayI begSegYDip,begSegXDip;
  TArrayI nsegYDip,nsegXDip;
  TArrayI segID;
  float *tmpSegZ,*tmpSegY,*tmpSegX;
  //
  // create segmentation in Z
  fNZSegDip = SegmentDimension(&tmpSegZ, fParamsDip, fNParamsDip, 2, 1,-1, 1,-1, 1,-1) - 1;
  fNYSegDip = 0;
  fNXSegDip = 0;
  //
  // for each Z slice create segmentation in Y
  begSegYDip.Set(fNZSegDip);
  nsegYDip.Set(fNZSegDip);
  float xyz[3];
  for (int iz=0;iz<fNZSegDip;iz++) {
    printf("\nZSegment#%d  %+e : %+e\n",iz,tmpSegZ[iz],tmpSegZ[iz+1]);
    int ny = SegmentDimension(&tmpSegY, fParamsDip, fNParamsDip, 1, 
				 1,-1, 1,-1, tmpSegZ[iz],tmpSegZ[iz+1]) - 1;
    segY.Set(ny + fNYSegDip);
    for (int iy=0;iy<ny;iy++) segY[fNYSegDip+iy] = tmpSegY[iy];
    begSegYDip[iz] = fNYSegDip;
    nsegYDip[iz] = ny;
    printf(" Found %d YSegments, to start from %d\n",ny, begSegYDip[iz]);
    //
    // for each slice in Z and Y create segmentation in X
    begSegXDip.Set(fNYSegDip+ny);
    nsegXDip.Set(fNYSegDip+ny);
    xyz[2] = (tmpSegZ[iz]+tmpSegZ[iz+1])/2.; // mean Z of this segment
    //
    for (int iy=0;iy<ny;iy++) {
      int isg = fNYSegDip+iy;
      printf("\n   YSegment#%d  %+e : %+e\n",iy, tmpSegY[iy],tmpSegY[iy+1]);
      int nx = SegmentDimension(&tmpSegX, fParamsDip, fNParamsDip, 0, 
				1,-1, tmpSegY[iy],tmpSegY[iy+1], tmpSegZ[iz],tmpSegZ[iz+1]) - 1;
      //
      segX.Set(nx + fNXSegDip);
      for (int ix=0;ix<nx;ix++) segX[fNXSegDip+ix] = tmpSegX[ix];
      begSegXDip[isg] = fNXSegDip;
      nsegXDip[isg] = nx;
      printf("   Found %d XSegments, to start from %d\n",nx, begSegXDip[isg]);
      //
      segID.Set(fNXSegDip+nx);
      //
      // find corresponding params
      xyz[1] = (tmpSegY[iy]+tmpSegY[iy+1])/2.; // mean Y of this segment
      //
      for (int ix=0;ix<nx;ix++) {
	xyz[0] = (tmpSegX[ix]+tmpSegX[ix+1])/2.; // mean X of this segment
	for (int ipar=0;ipar<fNParamsDip;ipar++) {
	  AliCheb3D* cheb = (AliCheb3D*) fParamsDip->At(ipar);
	  if (!cheb->IsInside(xyz)) continue;
	  segID[fNXSegDip+ix] = ipar;
	  break;
	}
      }
      fNXSegDip += nx;
      //
      delete[] tmpSegX;
    }
    delete[] tmpSegY;
    fNYSegDip += ny;
  }
  //
  fMinZDip = tmpSegZ[0];
  fMaxZDip = tmpSegZ[fNZSegDip];
  fSegZDip    = new Float_t[fNZSegDip];
  for (int i=fNZSegDip;i--;) fSegZDip[i] = tmpSegZ[i];
  delete[] tmpSegZ;
  //
  fSegYDip    = new Float_t[fNYSegDip];
  fSegXDip    = new Float_t[fNXSegDip];
  fBegSegYDip = new Int_t[fNZSegDip];
  fNSegYDip   = new Int_t[fNZSegDip];
  fBegSegXDip = new Int_t[fNYSegDip];
  fNSegXDip   = new Int_t[fNYSegDip];
  fSegIDDip   = new Int_t[fNXSegDip];
  //
  for (int i=fNYSegDip;i--;) fSegYDip[i] = segY[i];
  for (int i=fNXSegDip;i--;) fSegXDip[i] = segX[i];
  for (int i=fNZSegDip;i--;) {fBegSegYDip[i] = begSegYDip[i]; fNSegYDip[i] = nsegYDip[i];}
  for (int i=fNYSegDip;i--;) {fBegSegXDip[i] = begSegXDip[i]; fNSegXDip[i] = nsegXDip[i];}
  for (int i=fNXSegDip;i--;) {fSegIDDip[i]   = segID[i];}
  //
}
*/

//________________________________________________________________
void AliMagWrapCheb::SaveData(const char* outfile) const
{
  // writes coefficients data to output text file
  TString strf = outfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf.Data(),"w+");
  if (!stream) {
    AliErrorF("Failed to open output file %s",strf.Data());
    return;
  }
  //
  // Sol part    ---------------------------------------------------------
  fprintf(stream,"# Set of Chebyshev parameterizations for ALICE magnetic field\nSTART %s\n",GetName());
  fprintf(stream,"START SOLENOID\n#Number of pieces\n%d\n",fNParamsSol);
  for (int ip=0;ip<fNParamsSol;ip++) GetParamSol(ip)->SaveData(stream);
  fprintf(stream,"#\nEND SOLENOID\n");
  //
  // TPCInt part ---------------------------------------------------------
  //  fprintf(stream,"# Set of Chebyshev parameterizations for ALICE magnetic field\nSTART %s\n",GetName());
  fprintf(stream,"START TPCINT\n#Number of pieces\n%d\n",fNParamsTPC);
  for (int ip=0;ip<fNParamsTPC;ip++) GetParamTPCInt(ip)->SaveData(stream);
  fprintf(stream,"#\nEND TPCINT\n");
  //
  // TPCRatInt part ---------------------------------------------------------
  //  fprintf(stream,"# Set of Chebyshev parameterizations for ALICE magnetic field\nSTART %s\n",GetName());
  fprintf(stream,"START TPCRatINT\n#Number of pieces\n%d\n",fNParamsTPCRat);
  for (int ip=0;ip<fNParamsTPCRat;ip++) GetParamTPCRatInt(ip)->SaveData(stream);
  fprintf(stream,"#\nEND TPCRatINT\n");
  //
  // Dip part   ---------------------------------------------------------
  fprintf(stream,"START DIPOLE\n#Number of pieces\n%d\n",fNParamsDip);
  for (int ip=0;ip<fNParamsDip;ip++) GetParamDip(ip)->SaveData(stream);
  fprintf(stream,"#\nEND DIPOLE\n");
  //
  fprintf(stream,"#\nEND %s\n",GetName());
  //
  fclose(stream);
  //
}

Int_t AliMagWrapCheb::SegmentDimension(float** seg,const TObjArray* par,int npar, int dim, 
				       float xmn,float xmx,float ymn,float ymx,float zmn,float zmx)
{
  // find all boundaries in deimension dim for boxes in given region.
  // if mn>mx for given projection the check is not done for it.
  float *tmpC = new float[2*npar];
  int *tmpInd = new int[2*npar];
  int nseg0 = 0;
  for (int ip=0;ip<npar;ip++) {
    AliCheb3D* cheb = (AliCheb3D*) par->At(ip);
    if (xmn<xmx && (cheb->GetBoundMin(0)>(xmx+xmn)/2 || cheb->GetBoundMax(0)<(xmn+xmx)/2)) continue;
    if (ymn<ymx && (cheb->GetBoundMin(1)>(ymx+ymn)/2 || cheb->GetBoundMax(1)<(ymn+ymx)/2)) continue;
    if (zmn<zmx && (cheb->GetBoundMin(2)>(zmx+zmn)/2 || cheb->GetBoundMax(2)<(zmn+zmx)/2)) continue;
    //
    tmpC[nseg0++] = cheb->GetBoundMin(dim);
    tmpC[nseg0++] = cheb->GetBoundMax(dim);    
  }
  // range Dim's boundaries in increasing order
  TMath::Sort(nseg0,tmpC,tmpInd,kFALSE);
  // count number of really different Z's
  int nseg = 0;
  float cprev = -1e6;
  for (int ip=0;ip<nseg0;ip++) {
    if (TMath::Abs(cprev-tmpC[ tmpInd[ip] ])>1e-4) {
      cprev = tmpC[ tmpInd[ip] ];
      nseg++;
    }
    else tmpInd[ip] = -1; // supress redundant Z
  }
  // 
  *seg  = new float[nseg]; // create final Z segmenations
  nseg = 0;
  for (int ip=0;ip<nseg0;ip++) if (tmpInd[ip]>=0) (*seg)[nseg++] = tmpC[ tmpInd[ip] ];
  //
  delete[] tmpC;
  delete[] tmpInd;
  return nseg;
}

#endif

