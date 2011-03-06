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

    Double_t rowX = GetRowX( iRow );
	
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
    Double_t rowX = GetRowX( iRow );

    UInt_t ix0 = C.fP0 >>24;
    UInt_t ix1 = C.fP1 >>24;
    X = (float) ((ix1<<8) + ix0);
    Y = (float) (C.fP0 & 0x00FFFFFF);
    Z = (float) (C.fP1 & 0x00FFFFFF);
    X = (float) ((X - 32768.)*1.e-4 + rowX);
    Y = (float) ((Y - 8388608.)*1.e-4);
    Z = (float) ((Z - 8388608.)*1.e-4);
  }

  static float GetRowX( int iRow ){
    const float fgX[159] = 
      { 85.195f,
	85.945f,
	86.695f,
	87.445f,
	88.195f,
	88.945f,
	89.695f,
	90.445f,
	91.195f,
	91.945f,
	92.695f,
	93.445f,
	94.195f,
	94.945f,
	95.695f,
	96.445f,
	97.195f,
	97.945f,
	98.695f,
	99.445f,
	100.195f,
	100.945f,
	101.695f,
	102.445f,
	103.195f,
	103.945f,
	104.695f,
	105.445f,
	106.195f,
	106.945f,
	107.695f,
	108.445f,
	109.195f,
	109.945f,
	110.695f,
	111.445f,
	112.195f,
	112.945f,
	113.695f,
	114.445f,
	115.195f,
	115.945f,
	116.695f,
	117.445f,
	118.195f,
	118.945f,
	119.695f,
	120.445f,
	121.195f,
	121.945f,
	122.695f,
	123.445f,
	124.195f,
	124.945f,
	125.695f,
	126.445f,
	127.195f,
	127.945f,
	128.695f,
	129.445f,
	130.195f,
	130.945f,
	131.695f,
	135.180f,
	136.180f,
	137.180f,
	138.180f,
	139.180f,
	140.180f,
	141.180f,
	142.180f,
	143.180f,
	144.180f,
	145.180f,
	146.180f,
	147.180f,
	148.180f,
	149.180f,
	150.180f,
	151.180f,
	152.180f,
	153.180f,
	154.180f,
	155.180f,
	156.180f,
	157.180f,
	158.180f,
	159.180f,
	160.180f,
	161.180f,
	162.180f,
	163.180f,
	164.180f,
	165.180f,
	166.180f,
	167.180f,
	168.180f,
	169.180f,
	170.180f,
	171.180f,
	172.180f,
	173.180f,
	174.180f,
	175.180f,
	176.180f,
	177.180f,
	178.180f,
	179.180f,
	180.180f,
	181.180f,
	182.180f,
	183.180f,
	184.180f,
	185.180f,
	186.180f,
	187.180f,
	188.180f,
	189.180f,
	190.180f,
	191.180f,
	192.180f,
	193.180f,
	194.180f,
	195.180f,
	196.180f,
	197.180f,
	198.180f,
	199.430f,
	200.930f,
	202.430f,
	203.930f,
	205.430f,
	206.930f,
	208.430f,
	209.930f,
	211.430f,
	212.930f,
	214.430f,
	215.930f,
	217.430f,
	218.930f,
	220.430f,
	221.930f,
	223.430f,
	224.930f,
	226.430f,
	227.930f,
	229.430f,
	230.930f,
	232.430f,
	233.930f,
	235.430f,
	236.930f,
	238.430f,
	239.930f,
	241.430f,
	242.930f,
	244.430f,
	245.930f  };
    return fgX[iRow];
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
