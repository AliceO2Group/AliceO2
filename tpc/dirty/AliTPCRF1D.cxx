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

/* $Id$ */


//-----------------------------------------------------------------------------
//
//
//
//  Origin:   Marian Ivanov, Uni. of Bratislava, ivanov@fmph.uniba.sk
//
//  Declaration of class AliTPCRF1D
//
//-----------------------------------------------------------------------------

//

#include <RVersion.h>
#include <Riostream.h>
#include <TCanvas.h>
#include <TClass.h>
#include <TF2.h>
#include <TH1.h>
#include <TMath.h>
#include <TPad.h>
#include <TString.h>
#include <TStyle.h>

#include "AliTPCRF1D.h"

extern TStyle * gStyle; 

Int_t   AliTPCRF1D::fgNRF=100;  //default  number of interpolation points
Float_t AliTPCRF1D::fgRFDSTEP=0.01; //default step in cm

static Double_t funGauss(Double_t *x, Double_t * par)
{
  //Gauss function  -needde by the generic function object 
  return TMath::Exp(-(x[0]*x[0])/(2*par[0]*par[0]));
}

static Double_t funCosh(Double_t *x, Double_t * par)
{
  //Cosh function  -needde by the generic function object 
  return 1/TMath::CosH(3.14159*x[0]/(2*par[0]));  
}    

static Double_t funGati(Double_t *x, Double_t * par)
{
  //Gati function  -needde by the generic function object 
  Float_t k3=par[1];
  Float_t k3R=TMath::Sqrt(k3);
  Float_t k2=(TMath::Pi()/2)*(1-k3R/2.);
  Float_t k1=k2*k3R/(4*TMath::ATan(k3R));
  Float_t l=x[0]/par[0];
  Float_t tan2=TMath::TanH(k2*l);
  tan2*=tan2;
  Float_t res = k1*(1-tan2)/(1+k3*tan2);  
  return res;  
}    

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

ClassImp(AliTPCRF1D)


AliTPCRF1D::AliTPCRF1D(Bool_t direct,Int_t np,Float_t step)
           :TObject(),
	    fNRF(0),
            fDSTEPM1(0.),
            fcharge(0),
            forigsigma(0.),
            fpadWidth(3.5),
            fkNorm(0.5),
            fInteg(0.),
            fGRF(0),
            fSigma(0.),
            fOffset(0.),
            fDirect(kFALSE),
            fPadDistance(0.)
{
  //default constructor for response function object
  fDirect=direct;
  if (np!=0)fNRF = np;
  else (fNRF=fgNRF);
  fcharge = new Float_t[fNRF];
  if (step>0) fDSTEPM1=1./step;
  else fDSTEPM1 = 1./fgRFDSTEP;
  for(Int_t i=0;i<5;i++) {
    funParam[i]=0.;
    fType[i]=0;
  }
  
}

AliTPCRF1D::AliTPCRF1D(const AliTPCRF1D &prf)
           :TObject(prf),
	    fNRF(prf.fNRF),
            fDSTEPM1(prf.fDSTEPM1),
            fcharge(0),
            forigsigma(prf.forigsigma),
            fpadWidth(prf.fpadWidth),
            fkNorm(prf.fkNorm),
            fInteg(prf.fInteg),
            fGRF(new TF1(*(prf.fGRF))),
            fSigma(prf.fSigma),
            fOffset(prf.fOffset),
            fDirect(prf.fDirect),
            fPadDistance(prf.fPadDistance)
{
  //
  //
  for(Int_t i=0;i<5;i++) {
    funParam[i]=0.;
    fType[i]=0;
  }
  fcharge = new Float_t[fNRF];
  memcpy(fcharge,prf.fcharge, fNRF*sizeof(Float_t));

  //PH Change the name (add 0 to the end)
  TString s(fGRF->GetName());
  s+="0";
  fGRF->SetName(s.Data());
}

AliTPCRF1D & AliTPCRF1D::operator = (const AliTPCRF1D &prf)
{
  if(this!=&prf) {
    TObject::operator=(prf);
    fNRF=prf.fNRF;
    fDSTEPM1=prf.fDSTEPM1;
    delete [] fcharge;
    fcharge = new Float_t[fNRF];
    memcpy(fcharge,prf.fcharge, fNRF*sizeof(Float_t));
    forigsigma=prf.forigsigma;
    fpadWidth=prf.fpadWidth;
    fkNorm=prf.fkNorm;
    fInteg=prf.fInteg;
    delete fGRF;
    fGRF=new TF1(*(prf.fGRF));
   //PH Change the name (add 0 to the end)
    TString s(fGRF->GetName());
    s+="0";
    fGRF->SetName(s.Data());
    fSigma=prf.fSigma;
    fOffset=prf.fOffset;
    fDirect=prf.fDirect;
    fPadDistance=prf.fPadDistance;
  }
  return *this;
}



AliTPCRF1D::~AliTPCRF1D()
{
  //
  delete [] fcharge;
  delete fGRF;
}

Float_t AliTPCRF1D::GetRF(Float_t xin)
{
  //function which return response
  //for the charge in distance xin 
  //return linear aproximation of RF
  Float_t x = (xin-fOffset)*fDSTEPM1+fNRF/2;
  Int_t i1=Int_t(x);
  if (x<0) i1-=1;
  Float_t res=0;
  if (i1+1<fNRF &&i1>0)
    res = fcharge[i1]*(Float_t(i1+1)-x)+fcharge[i1+1]*(x-Float_t(i1));    
  return res;
}

Float_t  AliTPCRF1D::GetGRF(Float_t xin)
{  
  //function which returnoriginal charge distribution
  //this function is just normalised for fKnorm
  if (fGRF != 0 ) 
    return fkNorm*fGRF->Eval(xin)/fInteg;
      else
    return 0.;
}

   
void AliTPCRF1D::SetParam( TF1 * GRF,Float_t padwidth,
		       Float_t kNorm, Float_t sigma)
{
  //adjust parameters of the original charge distribution
  //and pad size parameters
   fpadWidth = padwidth;
   fGRF = GRF;
   fkNorm = kNorm;
   if (sigma==0) sigma= fpadWidth/TMath::Sqrt(12.);
   forigsigma=sigma;
   fDSTEPM1 = 10/TMath::Sqrt(sigma*sigma+fpadWidth*fpadWidth/12); 
   //sprintf(fType,"User");
   snprintf(fType,5,"User");
   //   Update();   
}
  

void AliTPCRF1D::SetGauss(Float_t sigma, Float_t padWidth,
		      Float_t kNorm)
{
  // 
  // set parameters for Gauss generic charge distribution
  //
  fpadWidth = padWidth;
  fkNorm = kNorm;
  if (fGRF !=0 ) fGRF->Delete();
  fGRF = new TF1("funGauss",funGauss,-5,5,1);
  funParam[0]=sigma;
  forigsigma=sigma;
  fGRF->SetParameters(funParam);
   fDSTEPM1 = 10./TMath::Sqrt(sigma*sigma+fpadWidth*fpadWidth/12); 
  //by default I set the step as one tenth of sigma  
  //sprintf(fType,"Gauss");
   snprintf(fType,5,"Gauss");
}

void AliTPCRF1D::SetCosh(Float_t sigma, Float_t padWidth,
		     Float_t kNorm)
{
  // 
  // set parameters for Cosh generic charge distribution
  //
  fpadWidth = padWidth;
  fkNorm = kNorm;
  if (fGRF !=0 ) fGRF->Delete();
  fGRF = new TF1("funCosh",	funCosh, -5.,5.,2);   
  funParam[0]=sigma;
  fGRF->SetParameters(funParam);
  forigsigma=sigma;
  fDSTEPM1 = 10./TMath::Sqrt(sigma*sigma+fpadWidth*fpadWidth/12); 
  //by default I set the step as one tenth of sigma
  //sprintf(fType,"Cosh");
  snprintf(fType,5,"Cosh");
}

void AliTPCRF1D::SetGati(Float_t K3, Float_t padDistance, Float_t padWidth,
		     Float_t kNorm)
{
  // 
  // set parameters for Gati generic charge distribution
  //
  fpadWidth = padWidth;
  fkNorm = kNorm;
  if (fGRF !=0 ) fGRF->Delete();
  fGRF = new TF1("funGati",	funGati, -5.,5.,2);   
  funParam[0]=padDistance;
  funParam[1]=K3;  
  fGRF->SetParameters(funParam);
  forigsigma=padDistance;
  fDSTEPM1 = 10./TMath::Sqrt(padDistance*padDistance+fpadWidth*fpadWidth/12); 
  //by default I set the step as one tenth of sigma
  //sprintf(fType,"Gati");
  snprintf(fType,5,"Gati");
}



void AliTPCRF1D::DrawRF(Float_t x1,Float_t x2,Int_t N)
{ 
  //
  //Draw prf in selected region <x1,x2> with nuber of diviision = n
  //
  char s[100];
  TCanvas  * c1 = new TCanvas("canRF","Pad response function",700,900);
  c1->cd();
  TPad * pad1 = new TPad("pad1RF","",0.05,0.55,0.95,0.95,21);
  pad1->Draw();
  TPad * pad2 = new TPad("pad2RF","",0.05,0.05,0.95,0.45,21);
  pad2->Draw();

  //sprintf(s,"RF response function for %1.2f cm pad width",
  //	  fpadWidth); 
  snprintf(s,60,"RF response function for %1.2f cm pad width",fpadWidth); 
  pad1->cd();
  TH1F * hRFo = new TH1F("hRFo","Original charge distribution",N+1,x1,x2);
  pad2->cd();
   gStyle->SetOptFit(1);
   gStyle->SetOptStat(0); 
  TH1F * hRFc = new TH1F("hRFc",s,N+1,x1,x2);
  Float_t x=x1;
  Float_t y1;
  Float_t y2;

  for (Float_t i = 0;i<N+1;i++)
    {
      x+=(x2-x1)/Float_t(N);
      y1 = GetRF(x);
      hRFc->Fill(x,y1);
      y2 = GetGRF(x);
      hRFo->Fill(x,y2);      
    };
  pad1->cd();
  hRFo->Fit("gaus");
  pad2->cd();
  hRFc->Fit("gaus");
}

void AliTPCRF1D::Update()
{
  //
  //update fields  with interpolated values for
  //PRF calculation

  //at the begining initialize to 0
  for (Int_t i =0; i<fNRF;i++)  fcharge[i] = 0;
  if ( fGRF == 0 ) return;
  // This form is no longer available 
#if ROOT_VERSION_CODE < ROOT_VERSION(5,99,0)
  fInteg  = fGRF->Integral(-5*forigsigma,5*forigsigma,funParam,0.00001);
#else
  TArrayD savParam(fGRF->GetNpar(), fGRF->GetParameters());
  fGRF->SetParameters(funParam);
  fInteg  = fGRF->Integral(-5*forigsigma,5*forigsigma,0.00001);
#endif
  if ( fInteg == 0 ) fInteg = 1; 
  if (fDirect==kFALSE){
  //integrate charge over pad for different distance of pad
  for (Int_t i =0; i<fNRF;i++)
    {      //x in cm fpadWidth in cm
      Float_t x = (Float_t)(i-fNRF/2)/fDSTEPM1;
      Float_t x1=TMath::Max(x-fpadWidth/2,-5*forigsigma);
      Float_t x2=TMath::Min(x+fpadWidth/2,5*forigsigma);
#if ROOT_VERSION_CODE < ROOT_VERSION(5,99,0)
      fcharge[i] = fkNorm*fGRF->Integral(x1,x2,funParam,0.0001)/fInteg;
#else
      fcharge[i] = fkNorm*fGRF->Integral(x1,x2,0.0001)/fInteg;
#endif
    };   
  }
  else{
    for (Int_t i =0; i<fNRF;i++)
      {      //x in cm fpadWidth in cm
	Float_t x = (Float_t)(i-fNRF/2)/fDSTEPM1;
	fcharge[i] = fkNorm*fGRF->Eval(x);
      };   
  }  
  fSigma = 0; 
  Float_t sum =0;
  Float_t mean=0;
  for (Float_t  x =-fNRF/fDSTEPM1; x<fNRF/fDSTEPM1;x+=1/fDSTEPM1)
    {      //x in cm fpadWidth in cm
      Float_t weight = GetRF(x+fOffset);
      fSigma+=x*x*weight; 
      mean+=x*weight;
      sum+=weight;
    };  
  if (sum>0){
    mean/=sum;
    fSigma = TMath::Sqrt(fSigma/sum-mean*mean);   
  }
  else fSigma=0; 
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
  fGRF->SetParameters(savParam.GetArray());
#endif
}

void AliTPCRF1D::Streamer(TBuffer &R__b)
{
   // Stream an object of class AliTPCRF1D.
   if (R__b.IsReading()) {
      AliTPCRF1D::Class()->ReadBuffer(R__b, this);
      //read functions
 
      if (strncmp(fType,"Gauss",3)==0) {delete fGRF; fGRF = new TF1("funGauss",funGauss,-5.,5.,4);}
      if (strncmp(fType,"Cosh",3)==0)  {delete fGRF; fGRF = new TF1("funCosh",funCosh,-5.,5.,4);}
      if (strncmp(fType,"Gati",3)==0)  {delete fGRF; fGRF = new TF1("funGati",funGati,-5.,5.,4);}  
      if (fGRF) fGRF->SetParameters(funParam);     

   } else {
      AliTPCRF1D::Class()->WriteBuffer(R__b, this);
   }
}


Double_t  AliTPCRF1D::Gamma4(Double_t x, Double_t p0, Double_t p1){
  //
  // Gamma 4 Time response function of ALTRO
  //
  if (x<0) return 0;
  Double_t g1 = TMath::Exp(-4.*x/p1);
  Double_t g2 = TMath::Power(x/p1,4);
  return p0*g1*g2;
}
 
