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

#include "Rtypes.h"
#include "TString.h"
#include "AliHLTTPCSpline2D3D.h"

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

  /** standard constructor */    
  AliHLTTPCFastTransform();           
  /** destructor */
  virtual ~AliHLTTPCFastTransform();
  
  /** initialization */
  Int_t  Init( AliTPCTransform *transform=0, Long_t TimeStamp=-1 );
 
  /** initialization */
  Bool_t IsInitialised() const { return fOrigTransform!=NULL; }
  
  /** deinitialization */
  void  DeInit();

  /** set the time stamp */
  Int_t SetCurrentTimeStamp( Long_t TimeStamp );

  /** Returns the current time stamp  */
  Long_t GetCurrentTimeStamp() const { return fLastTimeStamp; }

  /** Transformation: calibration + alignment */
  Int_t  Transform( Int_t Sector, Int_t Row, Float_t Pad, Float_t Time, Float_t XYZ[] );

  /** Transformation: calibration + alignment in double*/
  Int_t Transform( Int_t Sector, Int_t Row, Float_t Pad, Float_t Time, Double_t XYZ[] );

  /** Alignment */
  Int_t Alignment( Int_t iSec, Float_t XYZ[] );
  
  /** Alignment in double */
  Int_t Alignment( Int_t iSec, Double_t XYZ[] ); 

  /** Reverse alignment */
  Int_t ReverseAlignment( Int_t iSec, Float_t XYZ[] );

  /** Reverse alignment in double */
  Int_t ReverseAlignment( Int_t iSec, Double_t XYZ[] );

  /** Error string */
  const char* GetLastError() const { return fError.Data(); }

  /** total size of the object*/
  Int_t GetSize() const ;
  
  /** size of a particular row*/
  Int_t GetRowSize( Int_t iSec, Int_t iRow ) const;

  /** last calibrated time bin */
  Int_t GetLastTimeBin() const { return fLastTimeBin; }

  /** Print */
  void Print(const char* option=0) const;

 private:

  /** copy constructor prohibited */
  AliHLTTPCFastTransform(const AliHLTTPCFastTransform&);
  /** assignment operator prohibited */
  AliHLTTPCFastTransform& operator=(const AliHLTTPCFastTransform&);
  

 /** Initialisation of splines for a particular row */
  Int_t InitRow( Int_t iSector, Int_t iRow );

  /** Reverse rotation of matrix mA */
  bool CalcAdjugateRotation(const Float_t *mA, Float_t *mB, bool bCheck=0);
  
  struct AliRowTransform{
    AliHLTTPCSpline2D3D fSpline[3];
  };
 
  /** Set error string */
  Int_t Error(Int_t code, const char *msg);

  static const Int_t fkNSec = 72; //! transient
  static const Int_t fkNRows = 100; //! transient

  TString fError; // error string
  AliTPCTransform * fOrigTransform;                             //! transient
  Long_t fLastTimeStamp; // last time stamp
  Int_t fLastTimeBin; // last calibrated time bin
  Float_t fTimeBorder1; //! transient
  Float_t fTimeBorder2; //! transient
  Float_t *fAlignment; // alignment matrices translation,rotation,reverse rotation

  AliHLTTPCFastTransform::AliRowTransform *fRows[fkNSec][fkNRows]; //! transient

  ClassDef(AliHLTTPCFastTransform,0)
};

inline Int_t AliHLTTPCFastTransform::Transform( Int_t iSec, Int_t iRow, Float_t Pad, Float_t Time, Float_t XYZ[] ){
  if( fLastTimeStamp<0 || iSec<0 || iSec>=fkNSec || iRow<0 || iRow>=fkNRows || !fRows[iSec][iRow] ) return -1;
  Int_t iTime = ( Time>=fTimeBorder2 ) ?2 :( ( Time>fTimeBorder1 ) ?1 :0 );
  fRows[iSec][iRow]->fSpline[iTime].GetValue(Pad, Time, XYZ);              
  if( fAlignment ) Alignment( iSec, XYZ );
  return 0; 
}

inline Int_t  AliHLTTPCFastTransform::Transform( Int_t iSec, Int_t iRow, Float_t Pad, Float_t Time, Double_t XYZ[] ){
  if( fLastTimeStamp<0 || iSec<0 || iSec>=fkNSec || iRow<0 || iRow>=fkNRows || !fRows[iSec][iRow] ) return -1;
  Int_t iTime = ( Time>=fTimeBorder2 ) ?2 :( ( Time>fTimeBorder1 ) ?1 :0 );
  fRows[iSec][iRow]->fSpline[iTime].GetValue(Pad, Time, XYZ);              
  if( fAlignment ) Alignment( iSec, XYZ );
  return 0; 
}

inline Int_t AliHLTTPCFastTransform::Alignment( Int_t iSec, Float_t XYZ[] ){
  if( iSec<0 || iSec>=fkNSec ) return Error(-1, Form("AliHLTTPCFastTransform::Alignment: wrong sector %d", iSec));
  if( !fAlignment ) return 0;
  Float_t x=XYZ[0], y = XYZ[1], z = XYZ[2], *t = fAlignment + iSec*21, *r = t+3;
  XYZ[0] = t[0] + x*r[0] + y*r[1] + z*r[2];
  XYZ[1] = t[1] + x*r[3] + y*r[4] + z*r[5];
  XYZ[2] = t[2] + x*r[6] + y*r[7] + z*r[8];
  return 0;
}

inline Int_t AliHLTTPCFastTransform::Alignment( Int_t iSec, Double_t XYZ[] ){
  if( iSec<0 || iSec>=fkNSec ) return Error(-1, Form("AliHLTTPCFastTransform::Alignment: wrong sector %d", iSec));
  if( !fAlignment ) return 0;
  Float_t x=XYZ[0], y = XYZ[1], z = XYZ[2], *t = fAlignment + iSec*21, *r = t+3;
  XYZ[0] = t[0] + x*r[0] + y*r[1] + z*r[2];
  XYZ[1] = t[1] + x*r[3] + y*r[4] + z*r[5];
  XYZ[2] = t[2] + x*r[6] + y*r[7] + z*r[8];
  return 0;
}

inline Int_t AliHLTTPCFastTransform::ReverseAlignment( Int_t iSec, Float_t XYZ[] ){
  if( iSec<0 || iSec>=fkNSec ) return Error(-1, Form("AliHLTTPCFastTransform::ReverseAlignment: wrong sector %d", iSec));
  if( !fAlignment ) return 0;
  Float_t *t = fAlignment + iSec*21, *r = t+12, x=XYZ[0] - t[0], y = XYZ[1]-t[1], z = XYZ[2]-t[2];
  XYZ[0] = x*r[0] + y*r[1] + z*r[2];
  XYZ[1] = x*r[3] + y*r[4] + z*r[5];
  XYZ[2] = x*r[6] + y*r[7] + z*r[8];
  return 0;
}

inline Int_t AliHLTTPCFastTransform::ReverseAlignment( Int_t iSec, Double_t XYZ[] ){
  if( iSec<0 || iSec>=fkNSec ) return Error(-1, Form("AliHLTTPCFastTransform::ReverseAlignment: wrong sector %d", iSec));
  if( !fAlignment ) return 0;
  Float_t *t = fAlignment + iSec*21, *r = t+12, x=XYZ[0] - t[0], y = XYZ[1]-t[1], z = XYZ[2]-t[2];
  XYZ[0] = x*r[0] + y*r[1] + z*r[2];
  XYZ[1] = x*r[3] + y*r[4] + z*r[5];
  XYZ[2] = x*r[6] + y*r[7] + z*r[8];
  return 0;
}

inline Int_t AliHLTTPCFastTransform::Error(Int_t code, const char *msg)
{
  fError = msg;
  return code;
}

#endif
