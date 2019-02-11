/**************************************************************************
 * Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
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

/* $Id$ */

///////////////////////////////////////////////////////////////////
//                                                               //
// A straight line is coded as a point (3 Double_t) and           //
// 3 direction cosines                                           //
//                                                               //
///////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <TMath.h>

#include "AliStrLine.h"

using std::endl;
using std::cout;
ClassImp(AliStrLine)

//________________________________________________________
AliStrLine::AliStrLine() :
  TObject(),
  fWMatrix(0),
  fTpar(0)
 {
  // Default constructor
  for(Int_t i=0;i<3;i++) {
    fP0[i] = 0.;
    fSigma2P0[i] = 0.;
    fCd[i] = 0.;
  }
  SetIdPoints(65535,65535);
}

//________________________________________________________
AliStrLine::AliStrLine(const Double_t *const point, const Double_t *const cd, Bool_t twopoints, UShort_t id1, UShort_t id2) :
  TObject(),
  fWMatrix(0),
  fTpar(0)
{
  // Standard constructor
  // if twopoints is true:  point and cd are the 3D coordinates of
  //                        two points defininig the straight line
  // if twopoint is false: point represents the 3D coordinates of a point
  //                       belonging to the straight line and cd is the
  //                       direction in space
  for(Int_t i=0;i<3;i++)
    fSigma2P0[i] = 0.;

  if(twopoints)
    InitTwoPoints(point,cd);
  else 
    InitDirection(point,cd);

  SetIdPoints(id1,id2);
}


//________________________________________________________
AliStrLine::AliStrLine(const Double_t *const point, const Double_t *const sig2point, const Double_t *const cd, Bool_t twopoints, UShort_t id1, UShort_t id2) :
  TObject(),
  fWMatrix(0),
  fTpar(0)
{
  // Standard constructor
  // if twopoints is true:  point and cd are the 3D coordinates of
  //                        two points defininig the straight line
  // if twopoint is false: point represents the 3D coordinates of a point
  //                       belonging to the straight line and cd is the
  //                       direction in space
  for(Int_t i=0;i<3;i++)
    fSigma2P0[i] = sig2point[i];

  if(twopoints)
    InitTwoPoints(point,cd);
  else 
    InitDirection(point,cd);

  SetIdPoints(id1,id2);
}

//________________________________________________________
AliStrLine::AliStrLine(const Double_t *const point, const Double_t *const sig2point, const Double_t *const wmat, const Double_t *const cd, Bool_t twopoints, UShort_t id1, UShort_t id2) :
  TObject(),
  fWMatrix(0),
  fTpar(0)
{
  // Standard constructor
  // if twopoints is true:  point and cd are the 3D coordinates of
  //                        two points defininig the straight line
  // if twopoint is false: point represents the 3D coordinates of a point
  //                       belonging to the straight line and cd is the
  //                       direction in space
  Int_t k = 0;
  fWMatrix = new Double_t [6];
  for(Int_t i=0;i<3;i++){ 
    fSigma2P0[i] = sig2point[i];
    for(Int_t j=0;j<3;j++)if(j>=i)fWMatrix[k++]=wmat[3*i+j];
  }
  if(twopoints)
    InitTwoPoints(point,cd);
  else 
    InitDirection(point,cd);

  SetIdPoints(id1,id2);
}

//________________________________________________________
AliStrLine::AliStrLine(const AliStrLine &source):
  TObject(source),
  fWMatrix(0),
  fTpar(source.fTpar)
{
  //
  // copy constructor
  //
  for(Int_t i=0;i<3;i++){
    fP0[i]=source.fP0[i];
    fSigma2P0[i]=source.fSigma2P0[i];
    fCd[i]=source.fCd[i];
  }
  if(source.fWMatrix){
    fWMatrix = new Double_t [6];
    for(Int_t i=0;i<6;i++)fWMatrix[i]=source.fWMatrix[i];
  }
  for(Int_t i=0;i<2;i++) fIdPoint[i]=source.fIdPoint[i];
}

//________________________________________________________
AliStrLine& AliStrLine::operator=(const AliStrLine& source)
{
  // Assignment operator
  if(this !=&source){
    TObject::operator=(source);
    for(Int_t i=0;i<3;i++){
      fP0[i]=source.fP0[i];
      fSigma2P0[i]=source.fSigma2P0[i];
      fCd[i]=source.fCd[i];
    } 

    delete [] fWMatrix;
    fWMatrix=0;
    if(source.fWMatrix){
      fWMatrix = new Double_t [6];
      for(Int_t i=0;i<6;i++)fWMatrix[i]=source.fWMatrix[i];
    } 
    for(Int_t i=0;i<2;i++) fIdPoint[i]=source.fIdPoint[i];
  }
  return *this;
}

//________________________________________________________
void AliStrLine::GetWMatrix(Double_t *wmat)const {
// Getter for weighting matrix, as a [9] dim. array
  if(!fWMatrix)return;
  Int_t k = 0;
  for(Int_t i=0;i<3;i++){
    for(Int_t j=0;j<3;j++){
      if(j>=i){
	wmat[3*i+j]=fWMatrix[k++];
      }
      else{
	wmat[3*i+j]=wmat[3*j+i];
      }
    }
  }
} 

//________________________________________________________
void AliStrLine::SetWMatrix(const Double_t *wmat) {
// Setter for weighting matrix, strating from a [9] dim. array
  if(fWMatrix)delete [] fWMatrix;
  fWMatrix = new Double_t [6];
  Int_t k = 0;
  for(Int_t i=0;i<3;i++){
    for(Int_t j=0;j<3;j++)if(j>=i)fWMatrix[k++]=wmat[3*i+j];
  }
}

//________________________________________________________
void AliStrLine::InitDirection(const Double_t *const point, const Double_t *const cd)
{
  // Initialization from a point and a direction
  Double_t norm = cd[0]*cd[0]+cd[1]*cd[1]+cd[2]*cd[2];

  if(norm) {
    norm = TMath::Sqrt(1./norm);
    for(Int_t i=0;i<3;++i) {
      fP0[i]=point[i];
      fCd[i]=cd[i]*norm;
    }
    fTpar = 0.;
  }
  else AliFatal("Null direction cosines!!!");
}

//________________________________________________________
void AliStrLine::InitTwoPoints(const Double_t *const pA, const Double_t *const pB)
{
  // Initialization from the coordinates of two
  // points in the space
  Double_t cd[3];
  for(Int_t i=0;i<3;i++)cd[i] = pB[i]-pA[i];
  InitDirection(pA,cd);
}

//________________________________________________________
AliStrLine::~AliStrLine() {
  // destructor
  if(fWMatrix)delete [] fWMatrix;
}

//________________________________________________________
void AliStrLine::PrintStatus() const {
  // Print current status
  cout <<"=======================================================\n";
  cout <<"Direction cosines: ";
  for(Int_t i=0;i<3;i++)cout <<fCd[i]<<"; ";
  cout <<endl;
  cout <<"Known point: ";
  for(Int_t i=0;i<3;i++)cout <<fP0[i]<<"; ";
  cout <<endl;
  cout <<"Error on known point: ";
  for(Int_t i=0;i<3;i++)cout <<TMath::Sqrt(fSigma2P0[i])<<"; ";
  cout <<endl;
  cout <<"Current value for the parameter: "<<fTpar<<endl;
}

//________________________________________________________
Int_t AliStrLine::IsParallelTo(const AliStrLine *line) const {
  // returns 1 if lines are parallel, 0 if not paralel
  const Double_t prec=1e-14;
  Double_t cd2[3];
  line->GetCd(cd2);

  Double_t vecpx=fCd[1]*cd2[2]-fCd[2]*cd2[1];
  Double_t mod=TMath::Abs(fCd[1]*cd2[2])+TMath::Abs(fCd[2]*cd2[1]);
  if(TMath::Abs(vecpx) > prec*mod) return 0;

  Double_t vecpy=-fCd[0]*cd2[2]+fCd[2]*cd2[0];
  mod=TMath::Abs(fCd[0]*cd2[2])+TMath::Abs(fCd[2]*cd2[0]);
  if(TMath::Abs(vecpy) > prec*mod) return 0;

  Double_t vecpz=fCd[0]*cd2[1]-fCd[1]*cd2[0];
  mod=TMath::Abs(fCd[0]*cd2[1])+TMath::Abs(fCd[1]*cd2[0]);
  if(TMath::Abs(vecpz) > prec) return 0;

  return 1;
}
//________________________________________________________
Int_t AliStrLine::Crossrphi(const AliStrLine *line)
{
  // Cross 2 lines in the X-Y plane
  const Double_t prec=1e-14;
  const Double_t big=1e20;
  Double_t p2[3];
  Double_t cd2[3];
  line->GetP0(p2);
  line->GetCd(cd2);
  Double_t a=fCd[0];
  Double_t b=-cd2[0];
  Double_t c=p2[0]-fP0[0];
  Double_t d=fCd[1];
  Double_t e=-cd2[1];
  Double_t f=p2[1]-fP0[1];
  Double_t deno = a*e-b*d;
  Double_t mod=TMath::Abs(a*e)+TMath::Abs(b*d);
  Int_t retcode = 0;
  if(TMath::Abs(deno) > prec*mod) {
    fTpar = (c*e-b*f)/deno;
  }
  else {
    fTpar = big;
    retcode = -1;
  }
  return retcode;
}

//________________________________________________________
Int_t AliStrLine::CrossPoints(AliStrLine *line, Double_t *point1, Double_t *point2){
  // Looks for the crossing point estimated starting from the
  // DCA segment
  const Double_t prec=1e-14;
  Double_t p2[3];
  Double_t cd2[3];
  line->GetP0(p2);
  line->GetCd(cd2);
  Int_t i;
  Double_t k1 = 0;
  Double_t k2 = 0;
  Double_t a11 = 0;
  for(i=0;i<3;i++){
    k1+=(fP0[i]-p2[i])*fCd[i];
    k2+=(fP0[i]-p2[i])*cd2[i];
    a11+=fCd[i]*cd2[i];
  }
  Double_t a22 = -a11;
  Double_t a21 = 0;
  Double_t a12 = 0;
  for(i=0;i<3;i++){
    a21+=cd2[i]*cd2[i];
    a12-=fCd[i]*fCd[i];
  }
  Double_t deno = a11*a22-a21*a12;
  Double_t mod = TMath::Abs(a11*a22)+TMath::Abs(a21*a12);
  if(TMath::Abs(deno) < prec*mod) return -1;
  fTpar = (a11*k2-a21*k1) / deno;
  Double_t par2 = (k1*a22-k2*a12) / deno;
  line->SetPar(par2);
  GetCurrentPoint(point1);
  line->GetCurrentPoint(point2);
  return 0;
}
//________________________________________________________________
Int_t AliStrLine::Cross(AliStrLine *line, Double_t *point)
{

  //Finds intersection between lines
  Double_t point1[3];
  Double_t point2[3];
  Int_t retcod=CrossPoints(line,point1,point2);
  if(retcod==0){
    for(Int_t i=0;i<3;i++)point[i]=(point1[i]+point2[i])/2.;
    return 0;
  }else{
    return -1;
  }
}

//___________________________________________________________
Double_t AliStrLine::GetDCA(const AliStrLine *line) const
{
  //Returns the distance of closest approach between two lines
  const Double_t prec=1e-14;
  Double_t p2[3];
  Double_t cd2[3];
  line->GetP0(p2);
  line->GetCd(cd2);
  Int_t i;
  Int_t ispar=IsParallelTo(line);
  if(ispar){
    Double_t dist1q=0,dist2=0,mod=0;
    for(i=0;i<3;i++){
      dist1q+=(fP0[i]-p2[i])*(fP0[i]-p2[i]);
      dist2+=(fP0[i]-p2[i])*fCd[i];
      mod+=fCd[i]*fCd[i];
    }
    if(TMath::Abs(mod) > prec){
      dist2/=mod;
      return TMath::Sqrt(dist1q-dist2*dist2);
    }else{return -1;}
  }else{
     Double_t perp[3];
     perp[0]=fCd[1]*cd2[2]-fCd[2]*cd2[1];
     perp[1]=-fCd[0]*cd2[2]+fCd[2]*cd2[0];
     perp[2]=fCd[0]*cd2[1]-fCd[1]*cd2[0];
     Double_t mod=0,dist=0;
     for(i=0;i<3;i++){
       mod+=perp[i]*perp[i];
       dist+=(fP0[i]-p2[i])*perp[i];
     }
     if(TMath::Abs(mod) > prec) {
       return TMath::Abs(dist/TMath::Sqrt(mod));
     } else return -1;
  }
}
//________________________________________________________
void AliStrLine::GetCurrentPoint(Double_t *point) const {
  // Fills the array point with the current value on the line
  for(Int_t i=0;i<3;i++)point[i]=fP0[i]+fCd[i]*fTpar;
}

//________________________________________________________
Double_t AliStrLine::GetDistFromPoint(const Double_t *point) const 
{
  // computes distance from point 
  AliStrLine tmpline(point, fCd, kFALSE);
  return GetDCA(&tmpline);
}

//________________________________________________________
Bool_t AliStrLine::GetParamAtRadius(Double_t r,Double_t &t1,Double_t &t2) const
{
  // Input: radial distance from the origin (x=0, x=0) in the bending plane
  // Returns a boolean: kTRUE if the line crosses the cylinder of radius r
  // and axis coincident with the z axis. It returns kFALSE otherwise
  // Output: t1 and t2 in ascending order. The parameters of the line at 
  // the two intersections with the cylinder
  Double_t p1= fCd[0]*fP0[0]+fCd[1]*fP0[1];
  Double_t p2=fCd[0]*fCd[0]+fCd[1]*fCd[1];
  Double_t delta=p1*p1-p2*(fP0[0]*fP0[0]+fP0[1]*fP0[1]-r*r);
  if(delta<0.){
    t1=-1000.;
    t2=t1;
    return kFALSE;
  }
  delta=TMath::Sqrt(delta);
  t1=(-p1-delta)/p2;
  t2=(-p1+delta)/p2;

  if(t2<t1){
    // use delta as a temporary buffer
    delta=t1;
    t1=t2;
    t2=delta;
  }
  if(TMath::AreEqualAbs(t1,t2,1.e-9))t1=t2;
  return kTRUE;
}
