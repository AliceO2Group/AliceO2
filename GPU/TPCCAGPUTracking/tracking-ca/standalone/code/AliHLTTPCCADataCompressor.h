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
#include "AliHLTTPCCACompressedInputData.h"
#include "AliHLTTPCTransform.h"

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

  static AliHLTTPCCACompressedCluster PackXYZ( int iRow, float X, float Y, float Z )
  {
    // pack the cluster

    // get coordinates in [um]

    Double_t rowX = AliHLTTPCTransform::Row2X( iRow );
	
    Double_t x = (X - rowX )*1.e4 + 32768.;
    Double_t y = Y*1.e4 + 8388608.;
    Double_t z = Z*1.e4 + 8388608.;
	
    // truncate if necessary
    if( x<0 ) x = 0; else if( x > 0x0000FFFF ) x = 0x0000FFFF;
    if( y<0 ) y = 0; else if( y > 0x00FFFFFF ) y = 0x00FFFFFF;
    if( z<0 ) z = 0; else if( z > 0x00FFFFFF ) z = 0x00FFFFFF;
	
    UInt_t ix0 =  ( (UInt_t) x )&0x000000FF;
    UInt_t ix1 = (( (UInt_t) x )&0x0000FF00 )>>8;
    UInt_t iy = ( (UInt_t) y )&0x00FFFFFF;
    UInt_t iz = ( (UInt_t) z )&0x00FFFFFF;
	
    AliHLTTPCCACompressedCluster ret;
    ret.fP0 = (ix0<<24) + iy;
    ret.fP1 = (ix1<<24) + iz;      
    return ret;
  }

  static  void UnpackXYZ( int iRow, AliHLTTPCCACompressedCluster C, float &X, float &Y, float &Z  )
  {
    Double_t rowX = AliHLTTPCTransform::Row2X( iRow );

    UInt_t ix0 = C.fP0 >>24;
    UInt_t ix1 = C.fP1 >>24;
    X = (ix1<<8) + ix0;
    Y = C.fP0 & 0x00FFFFFF;
    Z = C.fP1 & 0x00FFFFFF;
    X = (X - 32768.)*1.e-4 + rowX;
    Y = (Y - 8388608.)*1.e-4;
    Z = (Z - 8388608.)*1.e-4;
  }

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

#endif //ALIHLTTPCCADATACOMPRESSOR_H
