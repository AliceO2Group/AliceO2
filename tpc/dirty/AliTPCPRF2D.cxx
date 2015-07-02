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

///////////////////////////////////////////////////////////////////////////////
//  AliTPCPRF2D -                                                            //
//  Pad response function object in two dimesions                            //
//  This class contains the basic functions for the                          //
//  calculation of PRF according generic charge distribution                 //
//  In Update function object calculate table of response function           //
//  in discrete x and y position                                             //
// This table is used for interpolation od response function in any position //
// (function GetPRF)                                                          //
//                                                                           // 
//  Origin: Marian Ivanov, Uni. of Bratislava, ivanov@fmph.uniba.sk          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <TCanvas.h>
#include <TClass.h>
#include <TF2.h>
#include <TH1.h> 
#include <TMath.h>
#include <TPad.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TText.h>
#include <string.h>

#include "AliH2F.h"
#include "AliTPCPRF2D.h"


extern TStyle * gStyle;

const Double_t AliTPCPRF2D::fgkDegtoRad = 0.01745329251994;
const Double_t AliTPCPRF2D::fgkSQRT12=3.464101;
const Int_t   AliTPCPRF2D::fgkNPRF = 100;


static Double_t FunGauss2D(const Double_t *const x, const Double_t *const par)
{ 
//Gauss function  -needde by the generic function object 
  return ( TMath::Exp(-(x[0]*x[0])/(2*par[0]*par[0]))*
	   TMath::Exp(-(x[1]*x[1])/(2*par[1]*par[1])));

}

static Double_t FunCosh2D(const Double_t *const x, const Double_t *const par)
{
 //Cosh function  -needde by the generic function object 
  return ( 1/(TMath::CosH(3.14159*x[0]/(2*par[0]))*
	   TMath::CosH(3.14159*x[1]/(2*par[1]))));
}    

static Double_t FunGati2D(const Double_t *const x, const Double_t *const par)
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
 //par[4] = is equal to k3Y
  k3=par[4];
  k3R=TMath::Sqrt(k3);
  k2=(TMath::Pi()/2)*(1-k3R/2.);
  k1=k2*k3R/(4*TMath::ATan(k3R));
  l=x[1]/par[0];
  tan2=TMath::TanH(k2*l); 
  tan2*=tan2;
  res = res*k1*(1-tan2)/(1+k3*tan2);   
  return res;  
}   

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

ClassImp(AliTPCPRF2D)

AliTPCPRF2D::AliTPCPRF2D()
            :TObject(),
             fcharge(0),
             fY1(0.),
             fY2(0.),
             fNYdiv(0),
             fNChargeArray(0),
             fChargeArray(0),
             fHeightFull(0.),
             fHeightS(0.),
             fShiftY(0.),
             fWidth(0.),
             fK(0.),
             fNPRF(0),
             fNdiv(5),
             fDStep(0.),
             fKNorm(1.),
             fInteg(0.),
             fGRF(0),
             fK3X(0.),
             fK3Y(0.),
             fPadDistance(0.),
             fOrigSigmaX(0.),
             fOrigSigmaY(0.),
             fChargeAngle(0.),
             fPadAngle(0.),
             fSigmaX(0.),
             fSigmaY(0.),
             fMeanX(0.),
             fMeanY(0.),
             fInterX(0),
             fInterY(0),
             fCurrentY(0.),
             fDYtoWire(0.),
             fDStepM1(0.)             
{
  //default constructor for response function object

  fNPRF =fgkNPRF ;
  for(Int_t i=0;i<5;i++){
        funParam[i]=0.;
        fType[i]=0;
  }
  

  //chewron default values   
  SetPad(0.8,0.8);
  SetChevron(0.2,0.0,1.0);
  SetY(-0.2,0.2,2);
  SetInterpolationType(2,0);
}

AliTPCPRF2D::~AliTPCPRF2D()
{
  if (fChargeArray!=0) delete [] fChargeArray;
  if (fGRF !=0 ) fGRF->Delete(); 
}

void AliTPCPRF2D::SetY(Float_t y1, Float_t y2, Int_t nYdiv)
{
  //
  //set virtual line position
  //first and last line and number of lines
  fNYdiv = nYdiv;
  fY1=y1;
  fY2=y2;
}

void AliTPCPRF2D::SetPad(Float_t width, Float_t height)
{
  //set base chevron parameters
 fHeightFull=height;
 fWidth=width;
}
void AliTPCPRF2D::SetChevron(Float_t hstep, 
			Float_t shifty, 
			Float_t fac)
{
  //set shaping of chewron parameters
  fHeightS=hstep;
  fShiftY=shifty;
  fK=fac;
}

void AliTPCPRF2D::SetChParam(Float_t width, Float_t height,
		  Float_t hstep, Float_t shifty, Float_t fac)
{
  SetPad(width,height);
  SetChevron(hstep,shifty,fac);
}


Float_t AliTPCPRF2D::GetPRF(Float_t xin, Float_t yin)
{
  //function which return pad response
  //for the charge in distance xin 
  //return  cubic aproximation of PRF or PRF at nearest virtual wire
   if (fChargeArray==0) return 0;
  //transform position to "wire position"
  Float_t y=fDYtoWire*(yin-fY1);
  if (fNYdiv == 1) y=fY1;
  //normaly it find nearest line charge
  if (fInterY ==0){   
    Int_t i=Int_t(0.5+y);
    if (y<0) i=Int_t(-0.5+y);
    if ((i<0) || (i>=fNYdiv) ) return 0;
    fcharge   = &(fChargeArray[i*fNPRF]);
    return GetPRFActiv(xin);
  }
  //make interpolation from more fore lines
  Int_t i= Int_t(y);
  Float_t res;
  if ((i<0) || (i>=fNYdiv) ) return 0;
  Float_t z0=0;
  Float_t z1=0;
  Float_t z2=0;
  Float_t z3=0;
  if (i>0) {
    fcharge =&(fChargeArray[(i-1)*fNPRF]);
    z0 = GetPRFActiv(xin);
  }
  fcharge =&(fChargeArray[i*fNPRF]);
  z1=GetPRFActiv(xin);
  if ((i+1)<fNYdiv){
    fcharge =&(fChargeArray[(i+1)*fNPRF]);
    z2 = GetPRFActiv(xin);
  }
  if ((i+2)<fNYdiv){
    fcharge =&(fChargeArray[(i+2)*fNPRF]);
    z3 = GetPRFActiv(xin);
  }
  Float_t a,b,c,d,k,l;
  a=z1;
  b=(z2-z0)/2.;
  k=z2-a-b;
  l=(z3-z1)/2.-b;
  d=l-2*k;
  c=k-d;
  Float_t dy=y-Float_t(i);
  
  res = a+b*dy+c*dy*dy+d*dy*dy*dy;  
  return res;            
} 


Float_t AliTPCPRF2D::GetPRFActiv(Float_t xin)
{
  //GEt response function on given charege line 
  //return spline aproximaton 
  Float_t x = (xin*fDStepM1)+fNPRF/2;
  Int_t i = Int_t(x);
  
  if  ( (i>1) && ((i+2)<fNPRF)) {
    Float_t a,b,c,d,k,l;
    a = fcharge[i];
    b = (fcharge[i+1]-fcharge[i-1])*0.5; 
    k = fcharge[i+1]-a-b;
    l = (fcharge[i+2]-fcharge[i])*0.5-b;
    d=l-2.*k;
    c=k-d;
    Float_t dx=x-Float_t(i);
    Float_t res = a+b*dx+c*dx*dx+d*dx*dx*dx;  
    return res;
  }
  else return 0;
}


Float_t  AliTPCPRF2D::GetGRF(Float_t xin, Float_t yin)
{  
  //function which returnoriginal charge distribution
  //this function is just normalised for fKnorm
  if (GetGRF() != 0 ) 
    return fKNorm*GetGRF()->Eval(xin,yin)/fInteg;
      else
    return 0.;
}

   
void AliTPCPRF2D::SetParam( TF2 *const GRF,  Float_t kNorm, 
		       Float_t sigmaX, Float_t sigmaY)
{
  //adjust parameters of the original charge distribution
  //and pad size parameters
   if (fGRF !=0 ) fGRF->Delete();
   fGRF = GRF;
   fKNorm = kNorm;
   //sprintf(fType,"User");
   snprintf(fType,5,"User");
   if (sigmaX ==0) sigmaX=(fWidth*(1+TMath::Abs(fK)))/fgkSQRT12;
   if (sigmaY ==0) sigmaY=(fWidth*(1+TMath::Abs(fK)))/fgkSQRT12;
   fOrigSigmaX=sigmaX; 
   fOrigSigmaY=sigmaY; 
   Double_t estimsigma = 
     TMath::Sqrt(sigmaX*sigmaX+(fWidth*fWidth*(1+TMath::Abs(fK))/12)+
		 TMath::Tan(fPadAngle*fgkDegtoRad)*TMath::Tan(fPadAngle*fgkDegtoRad)*fHeightFull*fHeightFull/12);   
   if (estimsigma < 5*sigmaX) {
     fDStep = estimsigma/10.;
     fNPRF  = Int_t(estimsigma*8./fDStep); 
   }
   else{
     fDStep = sigmaX; 
     Double_t width = fWidth*(1+TMath::Abs(fK))+TMath::Abs(TMath::Tan(fPadAngle*fgkDegtoRad))*fHeightFull;
     fNPRF = Int_t((width+8.*sigmaX)/fDStep);
   };

}
  

void AliTPCPRF2D::SetGauss(Float_t sigmaX, Float_t sigmaY,
		      Float_t kNorm)
{
  // 
  // set parameters for Gauss generic charge distribution
  //
  fKNorm = kNorm;
  fOrigSigmaX=sigmaX;
  fOrigSigmaY=sigmaY;
  //sprintf(fType,"Gauss");
  snprintf(fType,5,"Gauss");
  if (fGRF !=0 ) fGRF->Delete();
  fGRF = new TF2("FunGauss2D",FunGauss2D,-5.,5.,-5.,5.,4);
  
  funParam[0]=sigmaX;
  funParam[1]=sigmaY;  
  funParam[2]=fK;
  funParam[3]=fHeightS;    
 
  fGRF->SetParameters(funParam); 
  Double_t estimsigma = 
     TMath::Sqrt(sigmaX*sigmaX+(fWidth*fWidth*(1+TMath::Abs(fK))/12)+
		 TMath::Tan(fPadAngle)*TMath::Tan(fPadAngle*fgkDegtoRad)*fHeightFull*fHeightFull/12);   
   if (estimsigma < 5*sigmaX) {
     fDStep = estimsigma/10.;
     fNPRF  = Int_t(estimsigma*8./fDStep); 
   }
   else{
     fDStep = sigmaX; 
     Double_t width = fWidth*(1+TMath::Abs(fK))+TMath::Abs(TMath::Tan(fPadAngle*fgkDegtoRad))*fHeightFull;
     fNPRF = Int_t((width+8.*sigmaX)/fDStep);
   };
 
  
}
void AliTPCPRF2D::SetCosh(Float_t sigmaX, Float_t sigmaY,
		     Float_t kNorm)
{ 
  // set parameters for Cosh generic charge distribution
  //
  fKNorm = kNorm;
  fOrigSigmaX=sigmaX;
  fOrigSigmaY=sigmaY; 
  //  sprintf(fType,"Cosh");
  snprintf(fType,5,"Cosh");
  if (fGRF !=0 ) fGRF->Delete();
  fGRF = new TF2("FunCosh2D",	FunCosh2D,-5.,5.,-5.,5.,4);   
  funParam[0]=sigmaX;
  funParam[1]=sigmaY;
  funParam[2]=fK;  
  funParam[3]=fHeightS;
  fGRF->SetParameters(funParam);

  Double_t estimsigma = TMath::Sqrt(sigmaX*sigmaX+fWidth*fWidth*(1+TMath::Abs(fK))/12);   
  if (estimsigma < 5*sigmaX) {
    fDStep = estimsigma/10.;
    fNPRF  = Int_t(estimsigma*8./fDStep); 
  }
  else{
    fDStep = sigmaX; 
    fNPRF = Int_t((1.2*fWidth*(1+TMath::Abs(fK))+8.*sigmaX)/fDStep);
  };  
 
}

void AliTPCPRF2D::SetGati(Float_t K3X, Float_t K3Y,
		     Float_t padDistance,
		     Float_t kNorm)
{
  // set parameters for Gati generic charge distribution
  //
  fKNorm = kNorm;
  fK3X=K3X;
  fK3Y=K3Y;
  fPadDistance=padDistance;  
  //sprintf(fType,"Gati");
  snprintf(fType,5,"Gati");
  if (fGRF !=0 ) fGRF->Delete();
  fGRF = new TF2("FunGati2D",	FunGati2D,-5.,5.,-5.,5.,5);  
 
  funParam[0]=padDistance;
  funParam[1]=K3X;
  funParam[2]=fK;  
  funParam[3]=fHeightS;
  funParam[4]=K3Y;
  fGRF->SetParameters(funParam);
  fOrigSigmaX=padDistance;
  fOrigSigmaY=padDistance;
  Float_t sigmaX = fOrigSigmaX;
  Double_t estimsigma = TMath::Sqrt(sigmaX*sigmaX+fWidth*fWidth*(1+TMath::Abs(fK))/12);   
  if (estimsigma < 5*sigmaX) {
    fDStep = estimsigma/10.;
    fNPRF  = Int_t(estimsigma*8./fDStep); 
  }
  else{
    fDStep = sigmaX; 
    fNPRF = Int_t((1.2*fWidth*(1+TMath::Abs(fK))+8.*sigmaX)/fDStep);
  };
}



void AliTPCPRF2D::Update()
{
  //
  //update fields  with interpolated values for
  //PRF calculation

  if ( fGRF == 0 ) return;  
  //initialize interpolated values to 0
  Int_t i;
  if (fChargeArray!=0) delete [] fChargeArray;
  fChargeArray = new Float_t[fNPRF*fNYdiv];
  fNChargeArray = fNPRF*fNYdiv;
  for (i =0; i<fNPRF*fNYdiv;i++)  fChargeArray[i] = 0;
  //firstly calculate total integral of charge

  ////////////////////////////////////////////////////////
  //I'm waiting for normal integral
  //in this moment only sum
  Float_t x2=  4*fOrigSigmaX;
  Float_t y2=  4*fOrigSigmaY;
  Float_t dx = fOrigSigmaX/Float_t(fNdiv*6);
  Float_t dy = fOrigSigmaY/Float_t(fNdiv*6);  
  Int_t nx  = Int_t(0.5+x2/dx);
  Int_t ny  = Int_t(0.5+y2/dy);
  Int_t ix,iy;
  fInteg  = 0;
  Double_t dInteg =0;
  for (ix=-nx;ix<=nx;ix++)
    for ( iy=-ny;iy<=ny;iy++) 
      dInteg+=fGRF->Eval(Float_t(ix)*dx,Float_t(iy)*dy)*dx*dy;  
  /////////////////////////////////////////////////////
  fInteg =dInteg;
  if ( fInteg == 0 ) fInteg = 1; 

  for (i=0; i<fNYdiv; i++){
    if (fNYdiv == 1) fCurrentY = fY1;
    else
      fCurrentY = fY1+Double_t(i)*(fY2-fY1)/Double_t(fNYdiv-1);
    fcharge   = &(fChargeArray[i*fNPRF]);
    Update1();
  }
  //calculate conversion coefitient to convert position to virtual wire
  fDYtoWire=Float_t(fNYdiv-1)/(fY2-fY1);
  fDStepM1=1/fDStep;
  UpdateSigma();
}

void AliTPCPRF2D::Update1()
{
  //
  //update fields  with interpolated values for
  //PRF calculation for given charge line
  Int_t i;
  Double_t cos = TMath::Cos(fChargeAngle);
  Double_t sin = TMath::Sin(fChargeAngle);
  const Double_t kprec =0.00000001;
  //integrate charge over pad for different distance of pad
  for (i =0; i<fNPRF;i++){      
    //x in cm fWidth in cm
    //calculate integral 
    Double_t xch = fDStep * (Double_t)(i-fNPRF/2);
    fcharge[i]=0;
    Double_t k=1;  
    
    
    for (Double_t ym=-fHeightFull/2.-fShiftY;  ym<fHeightFull/2.-kprec;ym+=fHeightS){	
      Double_t y2chev=TMath::Min((ym+fHeightS),Double_t(fHeightFull/2.)); // end of chevron step
      Double_t y1chev= ym;  //beginning of chevron step
      Double_t y2 = TMath::Min(y2chev,fCurrentY+3.5*fOrigSigmaY);
      Double_t y1 = TMath::Max((y1chev),Double_t(-fHeightFull/2.));
      y1 = TMath::Max(y1chev,fCurrentY-3.5*fOrigSigmaY);

      Double_t x0 = fWidth*(-1.-(Double_t(k)*fK))*0.5+ym*TMath::Tan(fPadAngle*fgkDegtoRad);
      Double_t kx  = Double_t(k)*(fK*fWidth)/fHeightS;     
      kx = TMath::Tan(TMath::ATan(kx))+TMath::Tan(fPadAngle*fgkDegtoRad);     

      Int_t ny = TMath::Max(Int_t(fNdiv*TMath::Exp(-(y1-fCurrentY)*(y1-fCurrentY)/(2*fOrigSigmaY*fOrigSigmaY))),4);
      Double_t dy = TMath::Min(fOrigSigmaY/Double_t(ny),y2-y1);
      Double_t ndy = dy;
      
      //loop over different y strips with variable step size  dy
      if (y2>(y1+kprec)) for (Double_t y = y1; y<y2+kprec;){      
	//new step SIZE 
	
	ny = TMath::Max(Int_t(fNdiv*TMath::Exp(-(y-fCurrentY)*(y-fCurrentY)/(2*fOrigSigmaY*fOrigSigmaY))),5);
	ndy = fOrigSigmaY/Double_t(ny); 
	if (ndy>(y2-y-dy)) {
	  ndy =y2-y-dy;
	  if (ndy<kprec) ndy=2*kprec; //calculate new delta y
	}
	//		
	Double_t sumch=0;
	//calculation of x borders and initial step
	Double_t deltay = (y-y1chev);	 	

	Double_t xp1  = x0+deltay*kx;
                //x begining of pad at position y
	Double_t xp2 =xp1+fWidth;        //x end of pad at position y
	Double_t xp3 =xp1+kx*dy; //...at position y+dy
	Double_t xp4 =xp2+kx*dy; //..  
	
	Double_t x1 = TMath::Min(xp1,xp3);
	x1 = TMath::Max(xp1,xch-3.5*fOrigSigmaX); //beging of integration
	Double_t x2 = TMath::Max(xp2,xp4);
	 x2 = TMath::Min(xp2+dy*kx,xch+3.5*fOrigSigmaX); //end of integration

	Int_t nx = TMath::Max(Int_t(fNdiv*TMath::Exp(-(x1-xch)*(x1-xch)/(2*fOrigSigmaX*fOrigSigmaX))*
				    TMath::Exp(-(y1-fCurrentY)*(y1-fCurrentY)/(2*fOrigSigmaY*fOrigSigmaY))),2);
	Double_t dx = TMath::Min(fOrigSigmaX/Double_t(nx),x2-x1)/5.; //on the border more iteration
	Double_t ndx=dx;
	
	if (x2>(x1+kprec)) {
	  for (Double_t x = x1; x<x2+kprec ;){
	  //new step SIZE 	  
	  nx = TMath::Max(Int_t(fNdiv*TMath::Exp(-(x-xch)*(x-xch)/(2*fOrigSigmaX*fOrigSigmaX))),3);	  
	  ndx = fOrigSigmaX/Double_t(nx);
	  if (ndx>(x2-x-dx)) {
	    ndx =x2-x-dx;	 	   
	  }
          if ( ( (x+dx+ndx)<TMath::Max(xp3,xp1)) || ( (x+dx+ndx)>TMath::Min(xp4,xp2))) {
	    ndx/=5.;
	  }	  
	  if (ndx<kprec) ndx=2*kprec;
	  //INTEGRAL APROXIMATION
	  Double_t ddx,ddy,dddx,dddy;
	  ddx = xch-(x+dx/2.);
	  ddy = fCurrentY-(y+dy/2.);
	  dddx = cos*ddx-sin*ddy;
	  dddy = sin*ddx+cos*ddy;
	  Double_t z0=fGRF->Eval(dddx,dddy);  //middle point
	  
	  ddx = xch-(x+dx/2.);
	  ddy = fCurrentY-(y);
	  dddx = cos*ddx-sin*ddy;
	  dddy = sin*ddx+cos*ddy;
	  Double_t z1=fGRF->Eval(dddx,dddy);  //point down
	  
	  ddx = xch-(x+dx/2.);
	  ddy = fCurrentY-(y+dy);
	  dddx = cos*ddx-sin*ddy;
	  dddy = sin*ddx+cos*ddy;
	  Double_t z3=fGRF->Eval(dddx,dddy);  //point up
	  
	  ddx = xch-(x);
	  ddy = fCurrentY-(y+dy/2.);
	  dddx = cos*ddx-sin*ddy;
	  dddy = sin*ddx+cos*ddy;
	  Double_t z2=fGRF->Eval(dddx,dddy);  //point left  
	  
	  ddx = xch-(x+dx);
	  ddy = fCurrentY-(y+dy/2.);
	  dddx = cos*ddx-sin*ddy;
	  dddy = sin*ddx+cos*ddy;
	  Double_t z4=fGRF->Eval(dddx,dddy);  //point right
	  
	  
	  if (z0<0) {z0=0;z1=0;z2=0;z3=0;z4=0;}
	  
	  Double_t f2x= (z3+z1-2*z0)*4.;//second derivation in y
	  Double_t f2y= (z2+z4-2*z0)*4.;//second derivation in x
	  Double_t f1y= (z3-z1);
	  Double_t z ;	  
	  z = (z0+f2x/6.+f2y/6.);//second order aproxiation of integral	    
	  if (kx>kprec){  //positive derivation
	    if (x<(xp1+dy*kx)){                //calculate volume at left border 
	      Double_t xx1  = x;
	      Double_t xx2  = TMath::Min(x+dx,xp1+dy*kx);
	      Double_t yy1  = y+(xx1-xp1)/kx;
	      Double_t yy2  = TMath::Min(y+(xx2-xp1)/kx,y+dy);	      
	      z=z0;
	      if (yy2<y+dy) {		
		z-= z0*(y+dy-yy2)/dy; //constant part rectangle
		z-= f1y*(xx2-xx1)*(y+dy-yy2)*(y+dy-yy2)/(2.*dx*dy);
	      }
	      z-=z0*(xx2-xx1)*(yy2-yy1)/(2*dx*dy); //constant part rectangle
	      
	    }
	    if (x>xp2){          //calculate volume at right  border 
	      Double_t xx1  = x;
	      Double_t xx2  = x+dx;
	      Double_t yy1  = y+(xx1-xp2)/kx;
	      Double_t yy2  = y+(xx2-xp2)/kx;		     
	      z=z0;
	      //rectangle part
	      z-=z0*(yy1-y)/dy; //constant part
	      z-=f1y*(xx2-xx1)*(yy1-y)*(yy1-y)/(2*dx*dy);
	      //triangle part         
	      z-=z0*(xx2-xx1)*(yy2-yy1)/(2*dx*dy); //constant part	      
	    }
	  }	  
	  if (kx<-kprec){ //negative  derivation	    
	    if (x<(xp1+dy*kx)){       //calculate volume at left border          
	      Double_t xx1  = x;
	      Double_t xx2  = TMath::Min(x+dx,xp3-dy/kx);
	      Double_t yy1  = y+(xx1-xp1)/kx;
	      Double_t yy2  = TMath::Max(y,yy1+(xx2-xx1)/kx); //yy2<yy1 
	      z = z0;
	      z-= z0*(yy2-y)/dy; // constant part rectangle 
	      z-= f1y*(xx2-xx1)*(yy2-y)*(yy2-y)/(2.*dx*dy); 
	      z-=z0*(xx2-xx1)*(yy1-yy2)/(2*dx*dy); //constant part triangle
	    }
	    if (x>xp2){       //calculate volume at right  border 
	      Double_t xx1  = TMath::Max(x,xp2+dy*kx);
	      Double_t xx2  = x+dx;
	      Double_t yy1  = TMath::Min(y+dy,y-(xp2-xx1)/kx);
	      Double_t yy2  = y-(xp2-xx2)/kx;
	      z=z0;
	      z-=z0*(yy2-y)/dy;  //constant part rextangle
	      z-= f1y*(xx2-xx1)*(yy2-y)*(yy2-y)/(2.*dx*dy); 
	      z-=z0*(xx2-xx1)*(yy1-yy2)/(2*dx*dy); //constant part triangle
	    }     	    
	  }	
	       
	  if (z>0.)	      sumch+=fKNorm*z*dx*dy/fInteg;
	  
	  x+=dx;
	  dx = ndx;
	}; //loop over x  	  
	fcharge[i]+=sumch;
	}//if x2>x1
	y+=dy;
	dy =ndy;
      }//step over different y
      k*=-1.;
    }//step over chevron 
    
   }//step over different points on line NPRF
}

void AliTPCPRF2D::UpdateSigma()
{
  //
  //calulate effective sigma X and sigma y of PRF
  fMeanX = 0;
  fMeanY = 0;
  fSigmaX = 0;
  fSigmaY = 0;
 
  Float_t sum =0;
  Int_t i;
  Float_t x,y;

  for (i=-1; i<=fNYdiv; i++){
    if (fNYdiv == 1) y = fY1;
    else
      y = fY1+Float_t(i)*(fY2-fY1)/Float_t(fNYdiv-1);
    for (x =-fNPRF*fDStep; x<fNPRF*fDStep;x+=fDStep)
      {      
	//x in cm fWidth in cm
	Float_t weight = GetPRF(x,y);
	fSigmaX+=x*x*weight; 
	fSigmaY+=y*y*weight;
	fMeanX+=x*weight;
	fMeanY+=y*weight;
	sum+=weight;
    };  
  }
  if (sum>0){
    fMeanX/=sum;
    fMeanY/=sum;    
    fSigmaX = TMath::Sqrt(fSigmaX/sum-fMeanX*fMeanX);
    fSigmaY = TMath::Sqrt(fSigmaY/sum-fMeanY*fMeanY);   
  }
  else fSigmaX=0; 
}


void AliTPCPRF2D::Streamer(TBuffer &xRuub)
{
   // Stream an object of class AliTPCPRF2D

   if (xRuub.IsReading()) {
      UInt_t xRuus, xRuuc;
      Version_t xRuuv = xRuub.ReadVersion(&xRuus, &xRuuc);
      AliTPCPRF2D::Class()->ReadBuffer(xRuub, this, xRuuv, xRuus, xRuuc);
      //read functions
      if (strncmp(fType,"User",3)!=0){
	delete fGRF;  
        if (strncmp(fType,"Gauss",3)==0) 
	  fGRF = new TF2("FunGauss2D",FunGauss2D,-5.,5.,-5.,5.,4);
        if (strncmp(fType,"Cosh",3)==0) 
	  fGRF = new TF2("FunCosh2D",FunCosh2D,-5.,5.,-5.,5.,4);
        if (strncmp(fType,"Gati",3)==0) 
	  fGRF = new TF2("FunGati2D",FunGati2D,-5.,5.,-5.,5.,5);      
        if (fGRF!=0) fGRF->SetParameters(funParam);
      }
      //calculate conversion coefitient to convert position to virtual wire
      fDYtoWire=Float_t(fNYdiv-1)/(fY2-fY1);
      fDStepM1=1/fDStep;
   } else {
      AliTPCPRF2D::Class()->WriteBuffer(xRuub,this);
   }
}


TH1F *  AliTPCPRF2D::GenerDrawXHisto(Float_t x1, Float_t x2,Float_t y)
{
  //gener one dimensional hist of pad response function
  //  at position y 
  char s[100]; 
  const Int_t kn=200;
  //sprintf(s,"Pad Response Function"); 
  snprintf(s,100,"Pad Response Function");  
  TH1F * hPRFc = new TH1F("hPRFc",s,kn+1,x1,x2);
  Float_t x=x1;
  Float_t y1;

  for (Int_t i = 0;i<kn+1;i++)
    {
      x+=(x2-x1)/Float_t(kn);
      y1 = GetPRF(x,y);
      hPRFc->Fill(x,y1);
    };
  hPRFc->SetXTitle("pad  (cm)");
  return hPRFc;
}  

AliH2F * AliTPCPRF2D::GenerDrawHisto(Float_t x1, Float_t x2, Float_t y1, Float_t y2, Int_t Nx, Int_t Ny)
{
  //
  //gener two dimensional histogram with PRF
  //
  char s[100];
  //sprintf(s,"Pad Response Function"); 
  snprintf(s,100,"Pad Response Function");  
  AliH2F * hPRFc = new AliH2F("hPRFc",s,Nx,x1,x2,Ny,y1,y2);
  Float_t dx=(x2-x1)/Float_t(Nx);
  Float_t dy=(y2-y1)/Float_t(Ny) ;
  Float_t x,y,z; 
  x = x1;
  y = y1;
  for ( Int_t i  = 0;i<=Nx;i++,x+=dx){
    y=y1;
    for (Int_t j  = 0;j<=Ny;j++,y+=dy){
      z = GetPRF(x,y);
      hPRFc->SetBinContent(hPRFc->GetBin(i,j),z);
    };
  }; 
  hPRFc->SetXTitle("pad direction (cm)");
  hPRFc->SetYTitle("pad row  direction (cm)");
  hPRFc->SetTitleOffset(1.5,"X");
  hPRFc->SetTitleOffset(1.5,"Y");
  return hPRFc;
}


AliH2F * AliTPCPRF2D::GenerDrawDistHisto(Float_t x1, Float_t x2, Float_t y1, Float_t y2, Int_t Nx, Int_t Ny, Float_t  thr)
{
  //return histogram with distortion
  const Float_t kminth=0.00001;
  if (thr<kminth) thr=kminth;
  char s[100]; 
  //sprintf(s,"COG distortion of PRF (threshold=%2.2f)",thr); 
  snprintf(s,100,"COG distortion of PRF (threshold=%2.2f)",thr); 
  AliH2F * hPRFDist = new AliH2F("hDistortion",s,Nx,x1,x2,Ny,y1,y2);
  Float_t dx=(x2-x1)/Float_t(Nx);
  Float_t dy=(y2-y1)/Float_t(Ny) ;
  Float_t x,y,z,ddx;
  x=x1;
  for ( Int_t i  = 0;i<=Nx;i++,x+=dx){
    y=y1;
    for(Int_t j  = 0;j<=Ny;j++,y+=dy)      
      {
	Float_t sumx=0;
	Float_t sum=0;
	for (Int_t k=-3;k<=3;k++)
	  {	    
	    Float_t padx=Float_t(k)*fWidth;
	    z = GetPRF(x-padx,y); 
	    if (z>thr){
	      sum+=z;
	      sumx+=z*padx;
	    }	
	  };	
	if (sum>kminth)  
	  {
	    ddx = (x-(sumx/sum));
	  }
	else ddx=-1;
	if (TMath::Abs(ddx)<10) 	hPRFDist->SetBinContent(hPRFDist->GetBin(i,j),ddx);
      }
  }

  hPRFDist->SetXTitle("pad direction (cm)");
  hPRFDist->SetYTitle("pad row  direction (cm)");
  hPRFDist->SetTitleOffset(1.5,"X");
  hPRFDist->SetTitleOffset(1.5,"Y");
  return hPRFDist;
}  
  




void AliTPCPRF2D::DrawX(Float_t x1 ,Float_t x2,Float_t y1,Float_t y2, Int_t N)
{ 
  //
  //draw pad response function at interval <x1,x2> at  given y position
  //
  if (N<0) return;
  TCanvas  * c1 = new TCanvas("PRFX","Pad response function",700,900);
  c1->cd();  
  
  TPaveText * comment = new TPaveText(0.05,0.02,0.95,0.20,"NDC");
  comment->SetTextAlign(12);
  comment->SetFillColor(42);
  DrawComment(comment);  
  comment->Draw();
  c1->cd(); 

  TPad * pad2 = new TPad("pPRF","",0.05,0.22,0.95,0.95);
  pad2->Divide(2,(N+1)/2);
  pad2->Draw();
  gStyle->SetOptFit(1);
  gStyle->SetOptStat(1); 
  for (Int_t i=0;i<N;i++){
    char ch[200];
    Float_t y;
    if (N==1) y=y1;
    else y = y1+i*(y2-y1)/Float_t(N-1);
    pad2->cd(i+1);
    TH1F * hPRFc =GenerDrawXHisto(x1, x2,y);
    //sprintf(ch,"PRF at wire position: %2.3f",y);
    snprintf(ch,40,"PRF at wire position: %2.3f",y);
    hPRFc->SetTitle(ch);  
    //sprintf(ch,"PRF %d",i);
    snprintf(ch,15,"PRF %d",i);
    hPRFc->SetName(ch);  
     hPRFc->Fit("gaus");
  }
 
}



void AliTPCPRF2D::DrawPRF(Float_t x1 ,Float_t x2,Float_t y1, Float_t y2, Int_t Nx, Int_t Ny)
{ 
  //
  //
  TCanvas  * c1 = new TCanvas("canPRF","Pad response function",700,900);
  c1->cd();
  TPad * pad2 = new TPad("pad2PRF","",0.05,0.22,0.95,0.95);
  pad2->Draw(); 
  gStyle->SetOptFit(1);
  gStyle->SetOptStat(1); 
  TH2F * hPRFc = GenerDrawHisto(x1, x2, y1, y2, Nx,Ny);   
  pad2->cd();
  hPRFc->Draw("surf");
  c1->cd(); 
  TPaveText * comment = new TPaveText(0.05,0.02,0.95,0.20,"NDC");
  comment->SetTextAlign(12);
  comment->SetFillColor(42);
  DrawComment(comment);  
  comment->Draw();
}

void AliTPCPRF2D::DrawDist(Float_t x1 ,Float_t x2,Float_t y1, Float_t y2, Int_t Nx, Int_t Ny, Float_t thr)
{ 
  //
  //draw distortion of the COG method - for different threshold parameter
  TCanvas  * c1 = new TCanvas("padDistortion","COG distortion",700,900);
  c1->cd();
  TPad * pad1 = new TPad("dist","",0.05,0.55,0.95,0.95,21);
  pad1->Draw();
  TPad * pad2 = new TPad("dist","",0.05,0.22,0.95,0.53,21);
  pad2->Draw();
  gStyle->SetOptFit(1);
  gStyle->SetOptStat(0); 
  
  AliH2F * hPRFDist = GenerDrawDistHisto(x1, x2, y1, y2, Nx,Ny,thr); 
  
  pad1->cd();
  hPRFDist->Draw("surf");
  Float_t distmax =hPRFDist->GetMaximum();
  Float_t distmin =hPRFDist->GetMinimum();
  gStyle->SetOptStat(1); 
  
  TH1F * dist = hPRFDist->GetAmplitudes(distmin,distmax,distmin-1);
  pad2->cd();
  dist->Draw();
  c1->cd(); 
  TPaveText * comment = new TPaveText(0.05,0.02,0.95,0.20,"NDC");
  comment->SetTextAlign(12);
  comment->SetFillColor(42);
  DrawComment(comment);  
  comment->Draw();
}

void AliTPCPRF2D::DrawComment(TPaveText *comment)
{
  //
  //function to write comment to picture 
  
  char s[100];
  //draw comments to picture
  TText * title = comment->AddText("Pad Response Function  parameters:");
  title->SetTextSize(0.03);
  //sprintf(s,"Height of pad:  %2.2f cm",fHeightFull);
  snprintf(s,100,"Height of pad:  %2.2f cm",fHeightFull);
  comment->AddText(s);
  //sprintf(s,"Width pad:  %2.2f cm",fWidth);
  snprintf(s,100,"Width pad:  %2.2f cm",fWidth);
  comment->AddText(s);
  //sprintf(s,"Pad Angle:  %2.2f ",fPadAngle);
  snprintf(s,100,"Pad Angle:  %2.2f ",fPadAngle);
  comment->AddText(s);
  
  if (TMath::Abs(fK)>0.0001){
    //sprintf(s,"Height of one chevron unit h:  %2.2f cm",2*fHeightS);
    snprintf(s,100,"Height of one chevron unit h:  %2.2f cm",2*fHeightS);
    comment->AddText(s);
    //sprintf(s,"Overlap factor:  %2.2f",fK);
    snprintf(s,100,"Overlap factor:  %2.2f",fK);
    comment->AddText(s); 
  }

  if (strncmp(fType,"User",3)==0){
    //sprintf(s,"Charge distribution - user defined function  %s ",fGRF->GetTitle());
    snprintf(s,100,"Charge distribution - user defined function  %s ",fGRF->GetTitle());
    comment->AddText(s);  
    //sprintf(s,"Sigma x of charge distribution: %2.2f ",fOrigSigmaX);
    snprintf(s,100,"Sigma x of charge distribution: %2.2f ",fOrigSigmaX);  
    comment->AddText(s);  
    //sprintf(s,"Sigma y of charge distribution: %2.2f ",fOrigSigmaY);
    snprintf(s,100,"Sigma y of charge distribution: %2.2f ",fOrigSigmaY);
    comment->AddText(s); 
  }
  if (strncmp(fType,"Gauss",3)==0){
    //sprintf(s,"Gauss charge distribution");
    snprintf(s,100,"Gauss charge distribution");
    comment->AddText(s);  
    //sprintf(s,"Sigma x of charge distribution: %2.2f ",fOrigSigmaX);
    snprintf(s,100,"Sigma x of charge distribution: %2.2f ",fOrigSigmaX);
    comment->AddText(s);  
    //sprintf(s,"Sigma y of charge distribution: %2.2f ",fOrigSigmaY);
    snprintf(s,100,"Sigma y of charge distribution: %2.2f ",fOrigSigmaY);
    comment->AddText(s); 
  }
  if (strncmp(fType,"Gati",3)==0){
    //sprintf(s,"Gati charge distribution");
    snprintf(s,100,"Gati charge distribution");
    comment->AddText(s);  
    //sprintf(s,"K3X of Gati : %2.2f ",fK3X);
    snprintf(s,100,"K3X of Gati : %2.2f ",fK3X);
    comment->AddText(s);  
    //sprintf(s,"K3Y of Gati: %2.2f ",fK3Y);
    snprintf(s,100,"K3Y of Gati: %2.2f ",fK3Y);
    comment->AddText(s); 
    //sprintf(s,"Wire to Pad Distance: %2.2f ",fPadDistance);
    snprintf(s,100,"Wire to Pad Distance: %2.2f ",fPadDistance);
    comment->AddText(s); 
  }
  if (strncmp(fType,"Cosh",3)==0){
    //sprintf(s,"Cosh charge distribution");
    snprintf(s,100,"Cosh charge distribution");
    comment->AddText(s);  
    //sprintf(s,"Sigma x of charge distribution: %2.2f ",fOrigSigmaX);
    snprintf(s,100,"Sigma x of charge distribution: %2.2f ",fOrigSigmaX);
    comment->AddText(s);  
    //sprintf(s,"Sigma y of charge distribution: %2.2f ",fOrigSigmaY);
    snprintf(s,100,"Sigma y of charge distribution: %2.2f ",fOrigSigmaY);
    comment->AddText(s); 
  }
  //sprintf(s,"Normalisation: %2.2f ",fKNorm);
  snprintf(s,100,"Normalisation: %2.2f ",fKNorm);
  comment->AddText(s);    
}

