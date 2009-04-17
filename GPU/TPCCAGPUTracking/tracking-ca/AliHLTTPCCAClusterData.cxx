/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

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
