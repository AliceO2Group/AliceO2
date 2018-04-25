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
#include "AliHLTTPCReverseTransformInfoV1.h"

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

  /** N sectors */
  static Int_t GetNSec() { return fkNSec; }
  static Int_t GetNSecIn() { return fkNSecIn; }
  static Int_t GetNSecOut() { return fkNSecOut; }

  /** N rows per inner sector */
  static Int_t GetNRowsIn() { return fkNRowsIn; }

  /** N rows per outer sector */
  static Int_t GetNRowsOut() { return fkNRowsOut; }

  /** last calibrated time bin */
  Int_t GetLastTimeBin() const { return fLastTimeBin; }

  /** split of splines in time direction */
  Int_t GetTimeSplit1() const { return fTimeSplit1; }

  /** split of splines in time direction  */
  Int_t GetTimeSplit2() const { return fTimeSplit2; }

  /** alignment matrices*/
  const TArrayF & GetAlignment() const { return fAlignment; } 

  /** transformation spline */
  const AliHLTTPCSpline2D3DObject& GetSpline ( Int_t iSec, Int_t iRow, Int_t iSpline ) const { 
    if( iSec<fkNSecIn ) return fSplines[iSec*fkNRowsIn*3 + iRow*3 + iSpline];
    else return fSplines[fkNSplinesIn + (iSec-fkNSecIn)*fkNRowsOut*3 + iRow*3 + iSpline]; 
  }

  const AliHLTTPCSpline2D3DObject& GetSplineIn ( Int_t iSecIn, Int_t iRow, Int_t iSpline ) const { 
    return fSplines[iSecIn*fkNRowsIn*3 + iRow*3 + iSpline];
  }
 
  const AliHLTTPCSpline2D3DObject& GetSplineOut ( Int_t iSecOut, Int_t iRow, Int_t iSpline ) const { 
    return fSplines[fkNSplinesIn + iSecOut*fkNRowsOut*3 + iRow*3 + iSpline]; 
  }

  /** last calibrated time bin */
  void SetLastTimeBin( Int_t v ){ fLastTimeBin = v; }

  /** split of splines in time direction */
  void SetTimeSplit1( Float_t v){ fTimeSplit1 = v; }
 
  /** split of splines in time direction */
  void SetTimeSplit2( Float_t v){ fTimeSplit2 = v; }

  /** alignment matrices*/
  TArrayF & GetAlignmentNonConst(){ return fAlignment; } 

  /** transformation spline */
  AliHLTTPCSpline2D3DObject& GetSplineNonConst( Int_t iSec, Int_t iRow, Int_t iSpline ){ 
    if( iSec<fkNSecIn ) return fSplines[iSec*fkNRowsIn*3 + iRow*3 + iSpline];
    else return fSplines[fkNSplinesIn + (iSec-fkNSecIn)*fkNRowsOut*3 + iRow*3 + iSpline]; 
  }
  
  AliHLTTPCSpline2D3DObject& GetSplineInNonConst ( Int_t iSecIn, Int_t iRow, Int_t iSpline ) { 
    return fSplines[iSecIn*fkNRowsIn*3 + iRow*3 + iSpline];
  }
 
  AliHLTTPCSpline2D3DObject& GetSplineOutNonConst ( Int_t iSecOut, Int_t iRow, Int_t iSpline )  { 
    return fSplines[fkNSplinesIn + iSecOut*fkNRowsOut*3 + iRow*3 + iSpline]; 
  }

  bool IsSectorInit(int sec) const {return fSectorInit[sec];}
  void SetInitSec(Int_t sector, bool init) {if (sector >= 0 && sector < fkNSec) fSectorInit[sector] = init;}
  
  void Merge(const AliHLTTPCFastTransformObject& obj);
  Long64_t Merge(TCollection* list);
  
  void SetReverseTransformInfo(const AliHLTTPCReverseTransformInfoV1 p);
  AliHLTTPCReverseTransformInfoV1 GetReverseTransformInfo() const {return *(AliHLTTPCReverseTransformInfoV1*) fReverseTransformInfo;}

  private:
 
  static const Int_t fkNSecIn = 36; // transient
  static const Int_t fkNSecOut = 36; // transient
  static const Int_t fkNSec = 36+36;//fkNSecIn + fkNSecOut; 
  static const Int_t fkNRowsIn = 63; // transient
  static const Int_t fkNRowsOut = 96; // transient
  static const Int_t fkNSplinesIn = 36*63*3;// fkNSecIn*fkNRowsIn*3; 
  static const Int_t fkNSplinesOut = 36*96*3;// fkNSecOut*fkNRowsOut*3;

  bool fSectorInit[fkNSec];	//Sectors which are initialized


  Int_t fVersion;
  Int_t fLastTimeBin; // last calibrated time bin
  Float_t fTimeSplit1; // split of splines in time direction
  Float_t fTimeSplit2; // split of splines in time direction
  TArrayF fAlignment; // alignment matrices translation,rotation,reverse rotation
  AliHLTTPCSpline2D3DObject fSplines[fkNSplinesIn + fkNSplinesOut]; // transient

  Float_t fReverseTransformInfo[21]; // Object that contains reverse transform information, stored as float array to simplify root streaming

 public:

  ClassDef(AliHLTTPCFastTransformObject,1)
};

#endif
