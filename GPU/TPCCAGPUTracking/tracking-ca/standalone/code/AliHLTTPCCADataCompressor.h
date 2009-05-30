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

    GPUhd() static unsigned int IRowIClu2IDrc( unsigned int iRow, unsigned int iCluster ) {
      return ( iCluster << 8 ) + iRow;
    }

    GPUhd() static unsigned int IDrc2IRow( unsigned int IDrc ) { return ( IDrc % 256 ); }
    GPUhd() static unsigned int IDrc2IClu( unsigned int IDrc ) { return ( IDrc >> 8  ); }


    GPUhd() static unsigned int ISliceIRowIClu2IDsrc( unsigned int iSlice, unsigned int iRow, unsigned int iCluster ) {
      return ( iCluster << 14 ) + ( iRow << 6 ) + iSlice;
    }

    GPUhd() static unsigned int IDsrc2ISlice( unsigned int IDsrc ) {  return (  IDsrc % 64      ); }
    GPUhd() static unsigned int IDsrc2IRow  ( unsigned int IDsrc ) {  return ( ( IDsrc >> 6 ) % 256 ); }
    GPUhd() static unsigned int IDsrc2IClu  ( unsigned int IDsrc ) {  return (  IDsrc >> 14     ); }


    GPUhd() static unsigned short YZ2UShort( float Y, float Z );
    GPUhd() static float  UShort2Y ( unsigned short iYZ );
    GPUhd() static float  UShort2Z ( unsigned short iYZ );

};


// Inline methods


GPUhd() inline unsigned short AliHLTTPCCADataCompressor::YZ2UShort( float Y, float Z )
{
  // compress Y and Z coordinates in range [-3., 3.] to 16 bits

  const float kMult = 255. / 6.;
  Y = ( Y + 3.f ) * kMult;
  Z = ( Z + 3.f ) * kMult;
  if ( Y < 0. ) Y = 0.;
  else if ( Y > 255. ) Y = 255.;
  if ( Z < 0. ) Z = 0.;
  else if ( Z > 255. ) Z = 255.;
  return static_cast<unsigned short>( ( static_cast<unsigned int>( Y ) << 8 ) + static_cast<unsigned int>( Z ) );
}

GPUhd() inline float AliHLTTPCCADataCompressor::UShort2Y( unsigned short iYZ )
{
  // extract Y coordinate from the compressed 16bits format to [-3.,3.]

  const float kMult = 6.f / 255.f;
  return ( iYZ >> 8 )*kMult - 3.f;
}

GPUhd() inline float AliHLTTPCCADataCompressor::UShort2Z( unsigned short iYZ )
{
  // extract Z coordinate from the compressed 16bits format to [-3.,3.]

  const float kMult = 6.f / 255.f;
  return ( iYZ % 256 )*kMult - 3.f;
}

#endif
