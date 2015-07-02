#ifndef ALITPCRF1D_H
#define ALITPCRF1D_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

////////////////////////////////////////////////
//  Manager class for AliTPCRF1D              //
//////////////////////////////////////////////// 
  

// include files and class forward declarations


#include "TObject.h"
#include "TMath.h"
class TF1;


class AliTPCRF1D : public TObject {
public : 
  AliTPCRF1D(Bool_t direct=kFALSE,Int_t np=0,Float_t step=0 ); 
  AliTPCRF1D(const AliTPCRF1D &prf);
  AliTPCRF1D & operator = (const AliTPCRF1D &prf);
  ~AliTPCRF1D();  
  Float_t GetRF(Float_t xin); //return RF in point xin
  Float_t GetGRF(Float_t xin); //return generic response function  in xin
  void SetGauss(Float_t sigma,Float_t padWidth, Float_t kNorm);
  //adjust RF with GAUSIAN as generic GRF 
  //if  direct = kTRUE then it does't convolute distribution
  void SetCosh(Float_t sigma,Float_t padWidth, Float_t kNorm);
  void SetGati(Float_t K3, Float_t padDistance, Float_t padWidth,
	       Float_t kNorm);
  //adjust RF with 1/Cosh  as generic GRF
  void SetParam(TF1 * GRF,Float_t padwidth,Float_t kNorm, 
		Float_t sigma=0);
  //adjust RF with general function 
  void SetOffset(Float_t xoff) {fOffset=xoff;}
  //set offset value 
  Float_t GetOffset(){return fOffset;}
  Float_t GetPadWidth(){ return fpadWidth;};       
  //return  pad width 
  Float_t  GetSigma(){return fSigma;}
  //return estimated sigma of RF
  void DrawRF(Float_t x1=-3 ,Float_t x2 =3.,Int_t N = 200);
  //draw RF it don't delete histograms after drawing
  /// it's on user !!!!
  void Update();  
  static Double_t Gamma4(Double_t x, Double_t p0, Double_t p1);
private: 
  Double_t funParam[5];//parameters of used charge function
  Int_t  fNRF;      //number of interpolations point
  Float_t fDSTEPM1;    //element step for point
  Float_t* fcharge; //[fNPRF] field with RF
  Float_t  forigsigma;//sigma of original distribution;
  Float_t fpadWidth;  //width of pad
  Float_t fkNorm;     //normalisation factor of the charge integral
  Float_t fInteg;     //integral of GRF on +- infinity
  TF1 *  fGRF;        //charge distribution function
  Float_t fSigma;     //sigma of PAD response function

  Float_t fOffset;    //offset of response function (for time reponse we 
  //have for expample shifted gauss)
  //calculated during update
 
  Bool_t fDirect;     //tell us if we use directly generalfunction
  
  Float_t fPadDistance;   //pad to wire distance
  char  fType[5];     //type of the parametrisation
  static Int_t   fgNRF;//default  number of interpolation points
  static Float_t fgRFDSTEP;//default step in cm
  ClassDef(AliTPCRF1D,2)
}; 




#endif /* ALITPCRF1D_H */
  
