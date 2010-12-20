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
    X = (ix1<<8) + ix0;
    Y = C.fP0 & 0x00FFFFFF;
    Z = C.fP1 & 0x00FFFFFF;
    X = (X - 32768.)*1.e-4 + rowX;
    Y = (Y - 8388608.)*1.e-4;
    Z = (Z - 8388608.)*1.e-4;
  }

  static float GetRowX( int iRow ){
    const float fgX[159] = 
      { 85.195,
	85.945,
	86.695,
	87.445,
	88.195,
	88.945,
	89.695,
	90.445,
	91.195,
	91.945,
	92.695,
	93.445,
	94.195,
	94.945,
	95.695,
	96.445,
	97.195,
	97.945,
	98.695,
	99.445,
	100.195,
	100.945,
	101.695,
	102.445,
	103.195,
	103.945,
	104.695,
	105.445,
	106.195,
	106.945,
	107.695,
	108.445,
	109.195,
	109.945,
	110.695,
	111.445,
	112.195,
	112.945,
	113.695,
	114.445,
	115.195,
	115.945,
	116.695,
	117.445,
	118.195,
	118.945,
	119.695,
	120.445,
	121.195,
	121.945,
	122.695,
	123.445,
	124.195,
	124.945,
	125.695,
	126.445,
	127.195,
	127.945,
	128.695,
	129.445,
	130.195,
	130.945,
	131.695,
	135.180,
	136.180,
	137.180,
	138.180,
	139.180,
	140.180,
	141.180,
	142.180,
	143.180,
	144.180,
	145.180,
	146.180,
	147.180,
	148.180,
	149.180,
	150.180,
	151.180,
	152.180,
	153.180,
	154.180,
	155.180,
	156.180,
	157.180,
	158.180,
	159.180,
	160.180,
	161.180,
	162.180,
	163.180,
	164.180,
	165.180,
	166.180,
	167.180,
	168.180,
	169.180,
	170.180,
	171.180,
	172.180,
	173.180,
	174.180,
	175.180,
	176.180,
	177.180,
	178.180,
	179.180,
	180.180,
	181.180,
	182.180,
	183.180,
	184.180,
	185.180,
	186.180,
	187.180,
	188.180,
	189.180,
	190.180,
	191.180,
	192.180,
	193.180,
	194.180,
	195.180,
	196.180,
	197.180,
	198.180,
	199.430,
	200.930,
	202.430,
	203.930,
	205.430,
	206.930,
	208.430,
	209.930,
	211.430,
	212.930,
	214.430,
	215.930,
	217.430,
	218.930,
	220.430,
	221.930,
	223.430,
	224.930,
	226.430,
	227.930,
	229.430,
	230.930,
	232.430,
	233.930,
	235.430,
	236.930,
	238.430,
	239.930,
	241.430,
	242.930,
	244.430,
	245.930  };
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
