// @(#) $Id$
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

#include "AliHLTTPCCASliceOutput.h"

GPUhd() int AliHLTTPCCASliceOutput::EstimateSize( int nOfTracks, int nOfTrackClusters )
{
  // calculate the amount of memory [bytes] needed for the event

  const int kClusterDataSize = sizeof(  int ) + sizeof( unsigned short ) + sizeof( float2 ) + sizeof( float ) + sizeof( UChar_t ) + sizeof( UChar_t );

  return sizeof( AliHLTTPCCASliceOutput ) + sizeof( AliHLTTPCCASliceTrack )*nOfTracks + kClusterDataSize*nOfTrackClusters;
}

#ifndef HLTCA_GPUCODE
template<typename T> inline void AssignNoAlignment( T *&dst, char *&mem, int count )
{
  // assign memory to the pointer dst
  dst = ( T* ) mem;
  mem = ( char * )( dst + count );
}

GPUhd() void AliHLTTPCCASliceOutput::SetPointers()
{
  // set all pointers

  char *mem = &fMemory[0];
  AssignNoAlignment( fTracks,            mem, fNTracks );
  AssignNoAlignment( fClusterUnpackedYZ, mem, fNTrackClusters );
  AssignNoAlignment( fClusterUnpackedX,  mem, fNTrackClusters );
  AssignNoAlignment( fClusterId,         mem, fNTrackClusters );
  AssignNoAlignment( fClusterPackedYZ,   mem, fNTrackClusters );
  AssignNoAlignment( fClusterRow,        mem, fNTrackClusters );
  AssignNoAlignment( fClusterPackedAmp,  mem, fNTrackClusters );

}
#endif
