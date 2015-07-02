#ifndef ALITPCPRF2D_H
#define ALITPCPRF2D_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */
//////////////////////////////////////////////////////////////////
//  Manager class for AliTPCPRF2D                               //
//  This is to generate the 2-dimensional pad-response function //
//////////////////////////////////////////////////////////////////
#include "TObject.h"

class TF2;
class TArrayF;
class TH1F;
class AliH2F;
class TPaveText;

class AliTPCPRF2D : public TObject {
public : 
  AliTPCPRF2D();
  virtual ~AliTPCPRF2D(); 
  virtual void Update();  //recalculate tables for charge calculation
  Float_t GetGRF(Float_t xin, Float_t yin); 
  //return generic response function  in xin
  virtual TF2 * GetGRF(){return fGRF;}
  virtual Float_t GetPRF(Float_t xin, Float_t yin); 
  //return PRF in point xin,yin

  virtual void DrawX(Float_t x1 ,Float_t x2,Float_t y1,Float_t y2=0, Int_t N=1);
  virtual void DrawPRF(Float_t x1, Float_t x2, Float_t y1, Float_t y2, Int_t Nx=20, Int_t Ny=20);
  //draw two dimensional PRF

  virtual void DrawDist(Float_t x1, Float_t x2, Float_t y1, Float_t y2, Int_t Nx=20, Int_t Ny=20, 
		Float_t  thr=0);
  //draw distortion of COG method
  //we suppose threshold equal to thr
  TH1F *  GenerDrawXHisto(Float_t x1, Float_t x2,Float_t y);  
  AliH2F * GenerDrawHisto(Float_t x1, Float_t x2, Float_t y1, Float_t y2, Int_t Nx=20, Int_t Ny=20);
  AliH2F * GenerDrawDistHisto(Float_t x1, Float_t x2, Float_t y1, Float_t y2, Int_t Nx=20, Int_t Ny=20, 
		Float_t  thr=0);  
  
  virtual void SetPad(Float_t width, Float_t height);
  //set base chevron parameters
  virtual void SetChevron(Float_t hstep, Float_t shifty, Float_t fac);
  //set chevron parameters   
  virtual void SetChParam(Float_t width, Float_t height,
		  Float_t hstep, Float_t shifty, Float_t fac);
  //set all geometrical parameters     
  virtual void SetY(Float_t y1, Float_t y2, Int_t nYdiv) ;
  virtual void SetChargeAngle(Float_t angle){fChargeAngle = angle;} //set angle of pad and charge distribution
                                                            //axes
  virtual void SetCurrentAngle(Float_t /*angle*/){return;}
  virtual void SetPadAngle(Float_t angle){fPadAngle = angle;} //set pad angle
  void SetInterpolationType(Int_t interx, Int_t intery) {fInterX=interx; fInterY =intery;}
  virtual void SetGauss(Float_t sigmaX,Float_t sigmaY , Float_t kNorm=1);
  //adjust PRF with GAUSIAN as generic GRF 
  //if  direct = kTRUE then it does't convolute distribution
  virtual void SetCosh(Float_t sigmaX,Float_t sigmaY , Float_t kNorm=1);
  //adjust PRF with 1/Cosh  as generic GRF
  virtual void  SetGati(Float_t K3X, Float_t K3Y,
		     Float_t padDistance,
		     Float_t kNorm=1);
  void SetParam(TF2 *const GRF,Float_t kNorm, 
		Float_t sigmaX=0, Float_t sigmaY=0);
  void SetNdiv(Int_t Ndiv){fNdiv=Ndiv;}
  virtual Float_t GetSigmaX() const {return fSigmaX;}
  virtual Float_t GetSigmaY() const {return fSigmaY;}
  
  
protected:
  void Update1(); 
  virtual void UpdateSigma();  //recalculate sigma of PRF
  Float_t GetPRFActiv(Float_t xin); //return PRF in point xin and actual y
  Float_t  * fcharge; //!field with PRF 
  Float_t fY1;        //position of first "virtual" vire 
  Float_t fY2;        //position of last virtual vire
  Int_t fNYdiv;       //number of wires
  Int_t fNChargeArray;  //number of charge interpolation points
  Float_t * fChargeArray;  //[fNChargeArray]pointer to array of arrays
 
  void DrawComment(TPaveText * comment);  //draw comments to picture
  //chevron parameters
  Float_t fHeightFull;  //height of the full pad
  Float_t fHeightS;     //height of the one step
  Float_t fShiftY;      //shift of the step
  Float_t fWidth;       //width of the pad
  Float_t fK;           //k factor of the chewron

  Double_t funParam[5];//parameters of used charge function
  Int_t  fNPRF;      //number of interpolations point
  Int_t  fNdiv;      //number of division to calculate integral
  Float_t fDStep;    //element step for point 
  Float_t fKNorm;     //normalisation factor of the charge integral
  Float_t fInteg;     //integral of GRF on +- infinity
  TF2 *  fGRF;        //charge distribution function

  Float_t fK3X;       //KX parameter (only for Gati parametrization)
  Float_t fK3Y;       //KY parameter (only for Gati parametrisation)
  Float_t fPadDistance; //pad anode distnce (only for Gati parametrisation)

  Float_t  fOrigSigmaX; //sigma of original distribution;  
  Float_t  fOrigSigmaY; //sigma of original distribution;  

  Float_t  fChargeAngle;//'angle' of charge distribution refernce system to pad reference system
  Float_t  fPadAngle;   //'angle' of the pad assymetry

  Float_t  fSigmaX;    //sigma X of PAD response function
  Float_t  fSigmaY;    //sigma Y of PAD response function
  Float_t  fMeanX;     //mean X value
  Float_t  fMeanY;     //mean Y value
  Int_t    fInterX;    //interpolation in X
  Int_t    fInterY;    //interpolation in Y
  //calculated during update

  char  fType[5];       //charge type
  Float_t fCurrentY;    //in reality we calculate PRF only for one fixed y 
  Float_t fDYtoWire;    //! used to make PRF calculation faster in GetPRF
  Float_t fDStepM1;     //! used in GetPRFActiv to make calculation faster  
  //
  static const Double_t fgkDegtoRad; //numeric constant
  static const Double_t fgkSQRT12; //numeric constant
  static const Int_t   fgkNPRF;   //default number of division

private: 
  AliTPCPRF2D(const AliTPCPRF2D &prf);
  AliTPCPRF2D &operator = (const AliTPCPRF2D &/*prf*/) {return *this;}
  
  ClassDef(AliTPCPRF2D,1) 
};   

#endif /* ALITPCPRF2D_H */
