#ifndef MATERIALBUDGETMAP_H
#define MATERIALBUDGETMAP_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                                                                           //
//    Utility class to compute and draw Radiation Length Map                 //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
class TH2F;
class TClonesArray;
namespace o2
{
namespace steer
{
class MaterialBudgetMap : public TNamed  {

public:
  MaterialBudgetMap();
  MaterialBudgetMap(const char *title, Int_t mode, Int_t nc1,Float_t c1min,
	  Float_t c1max, Int_t nphi, Float_t phimin,
	  Float_t phimax,Float_t rmin,Float_t rmax,Float_t zmax);
  virtual ~MaterialBudgetMap();
  virtual void  Stepping();
  virtual void  BeginEvent();
  virtual void  FinishPrimary(Float_t c1, Float_t c2);
  virtual void  FinishEvent();
private:
   Int_t      mMode;             //!mode
   Float_t    mTotRadl;          //!Total Radiation length
   Float_t    mTotAbso;          //!Total absorption length
   Float_t    mTotGcm2;          //!Total g/cm2 traversed
   TH2F      *mHistRadl;         //!Radiation length map 
   TH2F      *mHistAbso;         //!Interaction length map
   TH2F      *mHistGcm2;         //!g/cm2 length map
   TH2F      *mHistReta;         //!Radiation length map as a function of eta
   TH2F      *mRZR;              //!Radiation lenghts at (R.Z)
   TH2F      *mRZA;              //!Absorbtion lengths at (R,Z)
   TH2F      *mRZG;              //!Density at (R,Z)
   Bool_t     mStopped;          //!Scoring has been stopped 
   Float_t    mRmin;             //!minimum radius
   Float_t    mZmax;             //!maximum radius
   Float_t    mRmax;             //!maximum z
};
}
}

#endif











