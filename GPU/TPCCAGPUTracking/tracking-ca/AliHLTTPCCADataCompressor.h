//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCADATACOMPRESSOR_H
#define ALIHLTTPCCADATACOMPRESSOR_H

#include "AliHLTTPCCADef.h"

/**
 * @class AliHLTTPCCADataCompressor
 *
 * The AliHLTTPCCADataCompressor class is used to
 * pack and unpack diffferent data, such as TPC cluster IDs, posistion, amplitude etc
 *
 */
class AliHLTTPCCADataCompressor
{
public:
  
  GPUhd() static UInt_t IRowIClu2IDrc( UInt_t iRow, UInt_t iCluster ){ 
    return (iCluster<<8)+iRow; 
  }

  GPUhd() static UInt_t IDrc2IRow( UInt_t IDrc ){ return ( IDrc%256 ); }  
  GPUhd() static UInt_t IDrc2IClu( UInt_t IDrc ){ return ( IDrc>>8  ); }  


  GPUhd() static UInt_t ISliceIRowIClu2IDsrc( UInt_t iSlice, UInt_t iRow, UInt_t iCluster ){ 
    return (iCluster<<14) + (iRow<<6) + iSlice; 
  }

  GPUhd() static UInt_t IDsrc2ISlice( UInt_t IDsrc ){  return (  IDsrc%64      ); }
  GPUhd() static UInt_t IDsrc2IRow  ( UInt_t IDsrc ){  return ( (IDsrc>>6)%256 ); }  
  GPUhd() static UInt_t IDsrc2IClu  ( UInt_t IDsrc ){  return (  IDsrc>>14     ); }  


  GPUhd() static UShort_t YZ2UShort( Float_t Y, Float_t Z );
  GPUhd() static Float_t  UShort2Y ( UShort_t iYZ );
  GPUhd() static Float_t  UShort2Z ( UShort_t iYZ );
  
};


// Inline methods


GPUhd() inline UShort_t AliHLTTPCCADataCompressor::YZ2UShort( Float_t Y, Float_t Z )
{ 
  // compress Y and Z coordinates in range [-3., 3.] to 16 bits

  const Float_t kMult = 255./6.;
  Y = (Y+3.)*kMult;
  Z = (Z+3.)*kMult;
  if( Y<0. ) Y = 0.;
  else if( Y>255. ) Y = 255.;
  if( Z<0. ) Z = 0.;
  else if( Z>255. ) Z = 255.;
  return static_cast<UShort_t>( ( static_cast<UInt_t>( Y )<<8) + static_cast<UInt_t>( Z ) );
}  

GPUhd() inline Float_t AliHLTTPCCADataCompressor::UShort2Y( UShort_t iYZ )
{ 
  // extract Y coordinate from the compressed 16bits format to [-3.,3.]

  const Float_t kMult = 6./255.; 
  return (iYZ >> 8)*kMult - 3.;
}  

GPUhd() inline Float_t AliHLTTPCCADataCompressor::UShort2Z( UShort_t iYZ )
{ 
  // extract Z coordinate from the compressed 16bits format to [-3.,3.]

  const Float_t kMult = 6./255.; 
  return (iYZ % 256)*kMult - 3.;
}

#endif
