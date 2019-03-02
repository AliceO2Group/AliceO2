// $Id$
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************



#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCAMath.h"

#ifndef assert
#include <assert.h>
#endif

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
#include <iostream>
#endif

GPUdi() void AliHLTTPCCAGrid::CreateEmpty()
{
  //Create an empty grid
  fYMin = 0.f;
  fYMax = 1.f;
  fZMin = 0.f;
  fZMax = 1.f;

  fNy = 0;
  fNz = 0;
  fN = 0;

  fStepYInv = 1.f;
  fStepZInv = 1.f;
}


GPUdi() void AliHLTTPCCAGrid::Create( float yMin, float yMax, float zMin, float zMax, float sy, float sz )
{
  //* Create the grid
  fYMin = yMin;
  fZMin = zMin;

  fStepYInv = 1.f / sy;
  fStepZInv = 1.f / sz;

  fNy = static_cast<unsigned int>( ( yMax - fYMin ) * fStepYInv + 1.f );
  fNz = static_cast<unsigned int>( ( zMax - fZMin ) * fStepZInv + 1.f );

  fN = fNy * fNz;

  fYMax = fYMin + fNy * sy;
  fZMax = fZMin + fNz * sz;
}


GPUdi() int AliHLTTPCCAGrid::GetBin( float Y, float Z ) const
{
  //* get the bin pointer
  const int yBin = static_cast<int>((Y - fYMin) * fStepYInv);
  const int zBin = static_cast<int>((Z - fZMin) * fStepZInv);
  const int bin = zBin * fNy + yBin;
#ifndef HLTCA_GPUCODE
  assert( bin >= 0 );
  assert( bin < static_cast<int>( fN ) );
#endif
  return bin;
}

GPUdi() int AliHLTTPCCAGrid::GetBinBounded( float Y, float Z ) const
{
  //* get the bin pointer
  const int yBin = static_cast<int>((Y - fYMin) * fStepYInv);
  const int zBin = static_cast<int>((Z - fZMin) * fStepZInv);
  const int bin = zBin * fNy + yBin;
  if ( bin < 0 ) return 0;
  if ( bin >= static_cast<int>( fN ) ) return fN - 1;
  return bin;
}

GPUdi() void AliHLTTPCCAGrid::GetBin( float Y, float Z, int* const bY, int* const bZ ) const
{
  //* get the bin pointer

  int bbY = ( int ) ( ( Y - fYMin ) * fStepYInv );
  int bbZ = ( int ) ( ( Z - fZMin ) * fStepZInv );

  if ( bbY < 0 ) bbY = 0;
  else if ( bbY >= ( int )fNy ) bbY = fNy - 1;
  if ( bbZ < 0 ) bbZ = 0;
  else if ( bbZ >= ( int )fNz ) bbZ = fNz - 1;
  *bY = ( unsigned int ) bbY;
  *bZ = ( unsigned int ) bbZ;
}

GPUdi() void AliHLTTPCCAGrid::GetBinArea( float Y, float Z, float dy, float dz, int& bin, int& ny, int& nz ) const
{
    Y -= fYMin;
    int by = (int) ((Y - dy) * fStepYInv);
    ny = (int) ((Y + dy) * fStepYInv) - by;
    Z -= fZMin;
    int bz = (int) ((Z - dz) * fStepZInv);
    nz = (int) ((Z + dz) * fStepZInv) - bz;
    if (by < 0) by = 0;
    else if (by >= (int) fNy) by = fNy - 1;
    if (bz < 0) bz = 0;
    else if (bz >= (int) fNz) bz = fNz - 1;
    if (by + ny >= (int) fNy) ny = fNy - 1 - by;
    if (bz + nz >= (int) fNz) nz = fNz - 1 - bz;
    bin = bz * fNy + by;
}
