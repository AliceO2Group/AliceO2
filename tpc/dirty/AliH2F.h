#ifndef ALIH2F_H
#define ALIH2F_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

// include files and class forward declarations

#include "TH2.h" 

class TH1F;
class TClonesArray;

class AliH2F : public TH2F {
public:
  AliH2F();
  AliH2F(const Text_t *name,const Text_t *title,Int_t nbinsx,
	     Axis_t xlow,Axis_t xup,Int_t nbinsy,Axis_t ylow,Axis_t yup);
    ~AliH2F();

public:
  AliH2F(const AliH2F &his);
  AliH2F & operator = (const AliH2F &his);
//  TClonesArray * FindPeaks(Float_t threshold, Float_t noise);  
    //find peaks and write it in form of AliTPCcluster to array
  void ClearSpectrum();
  void AddGauss(Float_t x,Float_t y,Float_t sx, Float_t sy, Float_t max);
  void AddNoise(Float_t sn);
  void ClearUnderTh(Int_t threshold);
  void Round();
  //round float values to integer values
 
  AliH2F * GetSubrange2d(Float_t xmin, Float_t xmax, 
			     Float_t ymin, Float_t ymax); 
  //create new  2D histogram 
  Float_t  GetOccupancy(Float_t th=1. , Float_t xmin=0, Float_t xmax=0, 
			     Float_t ymin=0, Float_t ymax=0);
  //calculate ration of channel over threshold to all channels
  TH1F * GetAmplitudes(Float_t zmin, Float_t zmax, Float_t th=1. , Float_t xmin=0, Float_t xmax=0, 
			     Float_t ymin=0, Float_t ymax=0);
		       //generate one dim histogram of amplitudes
 
public:  

protected:  

private:
 
  ClassDef(AliH2F,1)
};

#endif /*TH2FSMOOTH_H */
