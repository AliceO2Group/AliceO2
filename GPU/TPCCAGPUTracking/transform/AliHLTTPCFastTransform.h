#ifndef ALIHLTTPCFASTTRANSFORM_H
#define ALIHLTTPCFASTTRANSFORM_H

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

/** @file   AliHLTTPCFastTransform.h
    @author Sergey Gorbunov
    @date   
    @brief
*/

// see below for class documentation
// or
// refer to README to build package
// or
// visit http://web.ift.uib.no/~kjeks/doc/alice-hlt

#include"Rtypes.h"
#include"AliHLTTPCSpline2D3D.h"

class AliTPCTransform;

/**
 * @class AliHLTTPCFastTransform
 *
 * The class transforms internal TPC coordinates (pad,time) to XYZ. 
 * 
 * @ingroup alihlt_tpc_components
 */



class AliHLTTPCFastTransform{
    
 public:

  static AliHLTTPCFastTransform* Instance();
  static void Terminate();

  /** standard constructor */    
  AliHLTTPCFastTransform();           
  /** destructor */
  virtual ~AliHLTTPCFastTransform();
  
  /** initialization */
  Int_t  Init( AliTPCTransform *transform=0, Int_t TimeStamp=-1 );
  
  /** set the time stamp */
  void SetCurrentTimeStamp( Int_t TimeStamp );

  /** Transformation */
  Int_t  Transform( Int_t Sector, Int_t Row, Float_t Pad, Float_t Time, Float_t XYZ[] );

  /** Transformation in double*/
  Int_t Transform( Int_t Sector, Int_t Row, Float_t Pad, Float_t Time, Double_t XYZ[] );

  /** Initialisation of splines for a particular row */
  Int_t InitRow( Int_t iSector, Int_t iRow );

  /** total size of the object*/
  Int_t GetSize() const ;
  
  /** size of a particular row*/
  Int_t GetRowSize( Int_t iSec, Int_t iRow ) const;

  /** last calibrated time bin */
  Int_t GetLastTimeBin() const { return fLastTimeBin; }

 private:

  /** copy constructor prohibited */
  AliHLTTPCFastTransform(const AliHLTTPCFastTransform&);
  /** assignment operator prohibited */
  AliHLTTPCFastTransform& operator=(const AliHLTTPCFastTransform&);

  static AliHLTTPCFastTransform* fgInstance;  // singleton control

  struct AliRowTransform{
    AliHLTTPCSpline2D3D fSpline[3];
  };

  AliTPCTransform * fOrigTransform;                             //! transient
  Int_t fLastTimeStamp; // last time stamp
  Int_t fLastTimeBin; // last calibrated time bin
  Float_t fTimeBorder1; //! transient
  Float_t fTimeBorder2; //! transient

  AliHLTTPCFastTransform::AliRowTransform *fRows[72][100]; //! transient

  ClassDef(AliHLTTPCFastTransform,0)
};

inline Int_t AliHLTTPCFastTransform::Transform( Int_t iSec, Int_t iRow, Float_t Pad, Float_t Time, Float_t XYZ[] ){
  if( !fOrigTransform || iSec<0 || iSec>=72 || iRow<0 || iRow>=100 ) return 1;
  if( !fRows[iSec][iRow] && InitRow(iSec, iRow) ) return 1;
  Int_t iTime = ( Time>=fTimeBorder2 ) ?2 :( ( Time>fTimeBorder1 ) ?1 :0 );
  fRows[iSec][iRow]->fSpline[iTime].GetValue(Pad, Time, XYZ);              
  return 0; 
}

inline Int_t  AliHLTTPCFastTransform::Transform( Int_t iSec, Int_t iRow, Float_t Pad, Float_t Time, Double_t XYZ[] ){
  if( !fOrigTransform || iSec<0 || iSec>=72 || iRow<0 || iRow>=100 ) return 1;
  if( !fRows[iSec][iRow] && InitRow(iSec, iRow) ) return 1;
  Int_t iTime = ( Time>=fTimeBorder2 ) ?2 :( ( Time>fTimeBorder1 ) ?1 :0 );
  fRows[iSec][iRow]->fSpline[iTime].GetValue(Pad, Time, XYZ);              
  return 0; 
}


inline AliHLTTPCFastTransform* AliHLTTPCFastTransform::Instance(){ // Singleton implementation
  if( !fgInstance ) fgInstance = new AliHLTTPCFastTransform();  
  return fgInstance;
}

#endif
