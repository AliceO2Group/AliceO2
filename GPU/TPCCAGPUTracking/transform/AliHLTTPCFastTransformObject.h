#ifndef ALIHLTTPCFASTTRANSFORMOBJECT_H
#define ALIHLTTPCFASTTRANSFORMOBJECT_H

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

/** @file   AliHLTTPCFastTransformObjectObject.h
    @author Sergey Gorbunov
    @date   
    @brief
*/

// see below for class documentation
// or
// refer to README to build package
// or
// visit http://web.ift.uib.no/~kjeks/doc/alice-hlt

#include "TObject.h"
#include "AliHLTTPCSpline2D3DObject.h"

/**
 * The class to store transformation map for AliHLTTPCFastTransform
 */


class AliHLTTPCFastTransformObject: public TObject{
    
 public:

  /** standard constructor */    
  AliHLTTPCFastTransformObject();           

  /** constructor */
  AliHLTTPCFastTransformObject( const AliHLTTPCFastTransformObject & );

  /** assignment operator */
  AliHLTTPCFastTransformObject& operator=( const AliHLTTPCFastTransformObject &);

  /** destructor */
  ~AliHLTTPCFastTransformObject(){};

  /** version */
  Int_t GetVersion() const { return fVersion; }
 
  /** deinitialization */
  void  Reset();

  /** Max N sectors */
  static Int_t GetMaxSec() { return fkNSec; }

  /** Max N rows per sector */
  static Int_t GetMaxRows() { return fkNRows; }

  /** last calibrated time bin */
  Int_t GetLastTimeBin() const { return fLastTimeBin; }

  /** split of splines in time direction */
  Int_t GetTimeSplit1() const { return fTimeSplit1; }

  /** split of splines in time direction  */
  Int_t GetTimeSplit2() const { return fTimeSplit2; }

  /** alignment matrices*/
  const TArrayF & GetAlignment() const { return fAlignment; } 

  /** transformation spline */
  const AliHLTTPCSpline2D3DObject& GetSpline ( Int_t iSec, Int_t iRow, Int_t iSpline ) const { return fSplines[iSec*fkNRows*3 + iRow*3 + iSpline]; }

  /** last calibrated time bin */
  void SetLastTimeBin( Int_t v ){ fLastTimeBin = v; }

  /** split of splines in time direction */
  void SetTimeSplit1( Float_t v){ fTimeSplit1 = v; }
 
  /** split of splines in time direction */
  void SetTimeSplit2( Float_t v){ fTimeSplit2 = v; }

  /** alignment matrices*/
  TArrayF & GetAlignmentNonConst(){ return fAlignment; } 

  /** transformation spline */
  AliHLTTPCSpline2D3DObject& GetSplineNonConst( Int_t iSec, Int_t iRow, Int_t iSpline ){ return fSplines[iSec*fkNRows*3 + iRow*3 + iSpline]; }
  
  bool IsSectorInit(int sec) const {return fSectorInit[sec];}
  void SetInitSec(Int_t sector, bool init) {if (sector >= 0 && sector < fkNSec) fSectorInit[sector] = init;}
  
  void Merge(const AliHLTTPCFastTransformObject& obj);
  Long64_t Merge(TCollection* list);

  private:
 
  static const Int_t fkNSec = 72; // transient
  static const Int_t fkNRows = 100; // transient
  
  bool fSectorInit[fkNSec];	//Sectors which are initialized

  Int_t fVersion;
  Int_t fLastTimeBin; // last calibrated time bin
  Float_t fTimeSplit1; // split of splines in time direction
  Float_t fTimeSplit2; // split of splines in time direction
  TArrayF fAlignment; // alignment matrices translation,rotation,reverse rotation
  AliHLTTPCSpline2D3DObject fSplines[72*100*3]; // transient

 public:

  ClassDef(AliHLTTPCFastTransformObject,1)
};

#endif
