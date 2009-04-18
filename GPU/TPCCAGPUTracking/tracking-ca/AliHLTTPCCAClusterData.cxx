/**************************************************************************
 * This file is property of and copyright by the ALICE HLT Project        *
 * All rights reserved.                                                   *
 *                                                                        *
 * Primary Authors:                                                       *
 *     Copyright 2009       Matthias Kretz <kretz@kde.org>                *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCSpacePointData.h"
#include "AliHLTTPCCAMath.h"
#include <iostream>

void AliHLTTPCCAClusterData::readEvent( const AliHLTArray<AliHLTTPCSpacePointData *> &clusters,
                                        int numberOfClusters, double ClusterZCut )
{
  if ( numberOfClusters <= 0 ) {
    fSlice = -1;
    fFirstRow = 0;
    fLastRow = -1;
    return;
  }
  fSlice = clusters[0]->fID >> 25;
  fFirstRow = clusters[0]->fPadRow;
  fLastRow = fFirstRow;
  int row = fFirstRow;
  for ( int i = 0; i < row; ++i ) {
    fNumberOfClusters.push_back( 0 );
    fRowOffset.push_back( 0 );
  }
  fRowOffset.push_back( 0 );
  for ( int i = 0; i < numberOfClusters; ++i ) {
    AliHLTTPCSpacePointData *data = clusters[i];

    if ( CAMath::Abs( data->fZ ) > ClusterZCut ) continue;

    while ( row < data->fPadRow ) {
      fNumberOfClusters.push_back( fData.size() - fRowOffset.back() );
      fRowOffset.push_back( fData.size() );
      ++row;
    }
    Data d = { data->fX, data->fY, data->fZ, data->fID, data->fPadRow };
    fData.push_back( d );
  }
  fNumberOfClusters.push_back( fData.size() - fRowOffset.back() );
  fLastRow = row; // the last seen row is the last row in this slice
}

void AliHLTTPCCAClusterData::readEvent( const AliHLTTPCCAGBHit *hits, int *offset, int numberOfClusters )
{
  fSlice = hits[*offset].ISlice();
  fFirstRow = hits[*offset].IRow(); // the data is row sorted first in the slice, so this is our first row
  fLastRow = fFirstRow;
  int row = fFirstRow;
  for ( int i = 0; i < row; ++i ) {
    fNumberOfClusters.push_back( 0 );
    fRowOffset.push_back( 0 );
  }
  fRowOffset.push_back( 0 );
  for ( int &i = *offset; i < numberOfClusters; ++i ) {
    const AliHLTTPCCAGBHit &hit = hits[i];
    if ( hit.ISlice() != fSlice ) {
      // the data is slice sorted first so we're done gathering our data
      break;
    }
    while ( row < hit.IRow() ) {
      fNumberOfClusters.push_back( fData.size() - fRowOffset.back() );
      fRowOffset.push_back( fData.size() );
      ++row;
    }
    Data d = { hit.X(), hit.Y(), hit.Z(), hit.ID(), hit.IRow() };
    fData.push_back( d );
  }
  fNumberOfClusters.push_back( fData.size() - fRowOffset.back() );
  fLastRow = row; // the last seen row is the last row in this slice
}
