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

//----------------------------------------------------------------------------
//  Author:   Marian Ivanov
//
//  Implementation of class AliH2F
//
//-----------------------------------------------------------------------------

#include <TClonesArray.h>
#include <TMath.h>
#include <TRandom.h>

#include "AliH2F.h"


ClassImp(AliH2F)
//***********************************************************************
//***********************************************************************
//***********************************************************************
//***********************************************************************
AliH2F::AliH2F():TH2F() 
{
  //
 
}
AliH2F::AliH2F(const Text_t *name,const Text_t *title,
		       Int_t nbinsx,Axis_t xlow,Axis_t xup
		       ,Int_t nbinsy,Axis_t ylow,Axis_t yup):
  TH2F(name,title,nbinsx,xlow,xup
       ,nbinsy,ylow,yup)
{
  //
  
}
     
AliH2F::~AliH2F() 
{
  //
}

AliH2F::AliH2F(const AliH2F &his) :
  TH2F(his)
{
  //
  
}

AliH2F & AliH2F::operator = (const AliH2F & /*his*/) 
{
  //
  return *this;
}

/*
TClonesArray * AliH2F::FindPeaks(Float_t threshold, Float_t noise)
{
  //find peaks and write it in form of AliTPCcluster to array
    
  //firstly we need to create object for cluster finding
  //and fill it with contents of histogram
  AliTPCClusterFinder cfinder;
  cfinder.SetThreshold(threshold);
  cfinder.SetNoise(noise);
  cfinder.GetHisto(this);
  return cfinder.FindPeaks3();
}
*/

void AliH2F::ClearSpectrum()
{
  //clera histogram
  Int_t dimx =  fXaxis.GetNbins();
  Int_t dimy =  fYaxis.GetNbins();
  for (Int_t i = 0 ;i<dimx;i++)
    for (Int_t j = 0 ;j<dimy;j++) 
      {
	SetBinContent(GetBin(i,j),0);
	SetBinError(GetBin(i,j),0);
      }
}


void AliH2F::AddNoise(Float_t sn)
{
  // add gauss noise with sigma sn
  Int_t dimx =  fXaxis.GetNbins();
  Int_t dimy =  fYaxis.GetNbins();
  for (Int_t i = 0 ;i<dimx;i++)
    for (Int_t j = 0 ;j<dimy;j++) 
      {
        Float_t noise = gRandom->Gaus(0,sn);
	Float_t oldv  =GetBinContent(GetBin(i,j));
	Float_t olds  =GetBinError(GetBin(i,j));
	if (noise >0)
	  {
	    SetBinContent(GetBin(i,j),noise+oldv);
	    SetBinError(GetBin(i,j),TMath::Sqrt((noise*noise+olds*olds)));
	  }
      }
}
void AliH2F::AddGauss(Float_t x, Float_t y, 
			  Float_t sx, Float_t sy, Float_t max)
{  
  //transform to histogram coordinata  
  Int_t dimx =  fXaxis.GetNbins();
  Int_t dimy =  fYaxis.GetNbins();
  Float_t dx =(GetXaxis()->GetXmax()-GetXaxis()->GetXmin())/Float_t(dimx);
  Float_t dy =(GetYaxis()->GetXmax()-GetYaxis()->GetXmin())/Float_t(dimy);  
  //  x=(x-GetXaxis()->GetXmin())/dx;
  //y=(y-GetYaxis()->GetXmin())/dy;
  sx/=dx;
  sy/=dy;

  
  for (Int_t i = 0 ;i<dimx;i++)
    for (Int_t j = 0 ;j<dimy;j++) 
      {
	Float_t x2 =GetXaxis()->GetBinCenter(i+1);
	Float_t y2 =GetYaxis()->GetBinCenter(j+1);
	Float_t dx2 = (x2-x)*(x2-x);
        Float_t dy2 = (y2-y)*(y2-y);
        Float_t amp =max*exp(-(dx2/(2*sx*sx)+dy2/(2*sy*sy)));
	//Float_t oldv  =GetBinContent(GetBin(i+1,j+1));
	//	SetBinContent(GetBin(i+1,j+1),amp+oldv);
	Fill(x2,y2,amp);
      }
}

void AliH2F::ClearUnderTh(Int_t threshold)
{
  //clear histogram for bin under threshold
  Int_t dimx =  fXaxis.GetNbins();
  Int_t dimy =  fYaxis.GetNbins();
  for (Int_t i = 0 ;i<=dimx;i++)
    for (Int_t j = 0 ;j<=dimy;j++) 
      {	
	Float_t oldv  =GetBinContent(GetBin(i,j));
        if (oldv <threshold)
	  SetBinContent(GetBin(i,j),0);
      }
}

void AliH2F::Round()
{
  //round float to integer 
  Int_t dimx =  fXaxis.GetNbins();
  Int_t dimy =  fYaxis.GetNbins();
  for (Int_t i = 0 ;i<=dimx;i++)
    for (Int_t j = 0 ;j<=dimy;j++) 
      {	
	Float_t oldv  =GetBinContent(GetBin(i,j));
        oldv=(Int_t)oldv;
	SetBinContent(GetBin(i,j),oldv);
      }
}



AliH2F *AliH2F::GetSubrange2d(Float_t xmin, Float_t xmax, 
				      Float_t ymin, Float_t ymax)
{
  //this function return pointer to the new created 
  //histogram which is subhistogram of the 
  //calculate number
  //subhistogram range must be inside histogram

  if (xmax<=xmin) {
    xmin=fXaxis.GetXmin();
    xmax=fXaxis.GetXmax();
  }
  if (ymax<=ymin) {
     ymin=fYaxis.GetXmin();
     ymax=fYaxis.GetXmax();
  }

  Int_t nx = Int_t((xmax-xmin)/(fXaxis.GetXmax()-fXaxis.GetXmin())  * 
		   Float_t(fXaxis.GetNbins()));
  Int_t ny = Int_t((ymax-ymin)/(fYaxis.GetXmax()-fYaxis.GetXmin())  * 
		   Float_t(fYaxis.GetNbins()));
  TString  t1 = fName ;
  TString  t2 = fTitle ;
  t1+="_subrange";
  t2+="_subrange";
  const Text_t *ktt1 = t1;
  const Text_t *ktt2 = t2;
  
  AliH2F * sub = new AliH2F(ktt1,ktt2,nx,xmin,xmax,ny,ymin,ymax); 
  
  Int_t i1 = Int_t( Float_t(fXaxis.GetNbins())*(xmin-fXaxis.GetXmin())/
		    (fXaxis.GetXmax()-fXaxis.GetXmin()) ) ;
  Int_t i2 = Int_t( Float_t(fYaxis.GetNbins())*(ymin-fYaxis.GetXmin())/
		    (fYaxis.GetXmax()-fYaxis.GetXmin()) ) ;
  for (Int_t i=0;i<nx;i++)
    for (Int_t j=0;j<ny;j++)
      {
	Int_t index1 = GetBin(i1+i,i2+j);
	//        Int_t index2 = sub->GetBin(i,j);
        Float_t val = GetBinContent(index1);
	//        sub->SetBinContent(index2,val);
	//        Float_t err = GetBinError(index1);
        //sub->SetBinError(index2,GetBinError(index1));
        sub->SetBinContent(GetBin(i,j),val);
      }  
   return sub;
}

TH1F *AliH2F::GetAmplitudes(Float_t zmin, Float_t zmax, Float_t th, Float_t xmin, Float_t xmax, 
				      Float_t ymin, Float_t ymax)
{
  //this function return pointer to the new created 
  //histogram which is subhistogram of the 
  //calculate number
  //subhistogram range must be inside histogram
 
  if (xmax<=xmin) {
    xmin=fXaxis.GetXmin();
    xmax=fXaxis.GetXmax();
  }
  if (ymax<=ymin) {
     ymin=fYaxis.GetXmin();
     ymax=fYaxis.GetXmax();
  }
  Int_t nx = Int_t((xmax-xmin)/(fXaxis.GetXmax()-fXaxis.GetXmin())  * 
		   Float_t(fXaxis.GetNbins()));
  Int_t ny = Int_t((ymax-ymin)/(fYaxis.GetXmax()-fYaxis.GetXmin())  * 
		   Float_t(fYaxis.GetNbins()));
  TString  t1 = fName ;
  TString  t2 = fTitle ;
  t1+="_amplitudes";
  t2+="_amplitudes";
  const  Text_t *ktt1 = t1;
  const Text_t *ktt2 = t2;
  
  TH1F * h = new TH1F(ktt1,ktt2,100,zmin,zmax); 
  
  Int_t i1 = Int_t( Float_t(fXaxis.GetNbins())*(xmin-fXaxis.GetXmin())/
		    (fXaxis.GetXmax()-fXaxis.GetXmin()) ) ;
  Int_t i2 = Int_t( Float_t(fYaxis.GetNbins())*(ymin-fYaxis.GetXmin())/
		    (fYaxis.GetXmax()-fYaxis.GetXmin()) ) ;
  for (Int_t i=0;i<nx;i++)
    for (Int_t j=0;j<ny;j++)
      {
	Int_t index1 = GetBin(i1+i,i2+j);
        Float_t val = GetBinContent(index1);
        if (val>th) h->Fill(val);
      }  
   return h;
}

Float_t   AliH2F::GetOccupancy(Float_t th , Float_t xmin, Float_t xmax, 
			     Float_t ymin, Float_t ymax)
{
  //this function return pointer to the new created 
  //histogram which is subhistogram of the 
  //calculate number
  //subhistogram range must be inside histogram
 
  if (xmax<=xmin) {
    xmin=fXaxis.GetXmin();
    xmax=fXaxis.GetXmax();
  }
  if (ymax<=ymin) {
     ymin=fYaxis.GetXmin();
     ymax=fYaxis.GetXmax();
  }
  Int_t nx = Int_t((xmax-xmin)/(fXaxis.GetXmax()-fXaxis.GetXmin())  * 
		   Float_t(fXaxis.GetNbins()));
  Int_t ny = Int_t((ymax-ymin)/(fYaxis.GetXmax()-fYaxis.GetXmin())  * 
		   Float_t(fYaxis.GetNbins()));
 
  Int_t over =0; 
  Int_t i1 = Int_t( Float_t(fXaxis.GetNbins())*(xmin-fXaxis.GetXmin())/
		    (fXaxis.GetXmax()-fXaxis.GetXmin()) ) ;
  Int_t i2 = Int_t( Float_t(fYaxis.GetNbins())*(ymin-fYaxis.GetXmin())/
		    (fYaxis.GetXmax()-fYaxis.GetXmin()) ) ;
  for (Int_t i=0;i<nx;i++)
    for (Int_t j=0;j<ny;j++)
      {
	Int_t index1 = GetBin(i1+i,i2+j);
        Float_t val = GetBinContent(index1);
        if (val>th) over++;
      }  
  Int_t  all = nx*ny;
  if (all>0)  return Float_t(over)/Float_t(all);
  else 
    return 0;
}
