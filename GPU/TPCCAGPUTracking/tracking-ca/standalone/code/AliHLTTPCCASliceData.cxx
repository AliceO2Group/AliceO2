// **************************************************************************
// * This file is property of and copyright by the ALICE HLT Project        *
// * All rights reserved.                                                   *
// *                                                                        *
// * Primary Authors:                                                       *
// *     Copyright 2009       Matthias Kretz <kretz@kde.org>                *
// *                                                                        *
// * Permission to use, copy, modify and distribute this software and its   *
// * documentation strictly for non-commercial purposes is hereby granted   *
// * without fee, provided that the above copyright notice appears in all   *
// * copies and that both the copyright notice and this permission notice   *
// * appear in the supporting documentation. The authors make no claims     *
// * about the suitability of this software for any purpose. It is          *
// * provided "as is" without express or implied warranty.                  *
// **************************************************************************

#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTArray.h"
#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCAParam.h"
#include "MemoryAssignmentHelpers.h"
#include "AliHLTTPCCAGPUConfig.h"
#include "AliHLTTPCCAGPUTracker.h"
#include <iostream>

// calculates an approximation for 1/sqrt(x)
// Google for 0x5f3759df :)
static inline float fastInvSqrt( float _x )
{
  // the function calculates fast inverse sqrt

  union { float f; int i; } x = { _x };
  const float xhalf = 0.5f * x.f;
  x.i = 0x5f3759df - ( x.i >> 1 );
  x.f = x.f * ( 1.5f - xhalf * x.f * x.f );
  return x.f;
}

inline void AliHLTTPCCASliceData::CreateGrid( AliHLTTPCCARow *row, const AliHLTTPCCAClusterData &data, int ClusterDataHitNumberOffset )
{
  // grid creation

  if ( row->NHits() <= 0 ) { // no hits or invalid data
    // grid coordinates don't matter, since there are no hits
    row->fGrid.CreateEmpty();
    return;
  }

  float yMin =  1.e3f;
  float yMax = -1.e3f;
  float zMin =  1.e3f;
  float zMax = -1.e3f;
  for ( int i = ClusterDataHitNumberOffset; i < ClusterDataHitNumberOffset + row->fNHits; ++i ) {
    const float y = data.Y( i );
    const float z = data.Z( i );
    if ( yMax < y ) yMax = y;
    if ( yMin > y ) yMin = y;
    if ( zMax < z ) zMax = z;
    if ( zMin > z ) zMin = z;
  }

  const float norm = fastInvSqrt( row->fNHits );
  row->fGrid.Create( yMin, yMax, zMin, zMax,
                     CAMath::Max( ( yMax - yMin ) * norm, 2.f ),
                     CAMath::Max( ( zMax - zMin ) * norm, 2.f ) );
}

inline void AliHLTTPCCASliceData::PackHitData( AliHLTTPCCARow* const row, const AliHLTArray<AliHLTTPCCAHit> &binSortedHits )
{
  // hit data packing

  static const float shortPackingConstant = 1.f / 65535.f;
  const float y0 = row->fGrid.YMin();
  const float z0 = row->fGrid.ZMin();
  const float stepY = ( row->fGrid.YMax() - y0 ) * shortPackingConstant;
  const float stepZ = ( row->fGrid.ZMax() - z0 ) * shortPackingConstant;
  const float stepYi = 1.f / stepY;
  const float stepZi = 1.f / stepZ;

  row->fHy0 = y0;
  row->fHz0 = z0;
  row->fHstepY = stepY;
  row->fHstepZ = stepZ;
  row->fHstepYi = stepYi;
  row->fHstepZi = stepZi;

  for ( int hitIndex = 0; hitIndex < row->fNHits; ++hitIndex ) {
    // bin sorted index!
    const int globalHitIndex = row->fHitNumberOffset + hitIndex;
    const AliHLTTPCCAHit &hh = binSortedHits[globalHitIndex];
    const float xx = ( ( hh.Y() - y0 ) * stepYi ) + .5 ;
    const float yy = ( ( hh.Z() - z0 ) * stepZi ) + .5 ;
    if ( xx < 0 || yy < 0 || xx >= 65536  || yy >= 65536 ) {
      std::cout << "!!!! hit packing error!!! " << xx << " " << yy << " " << std::endl;
    }
    // HitData is bin sorted
    fHitData[row->fHitNumberOffset + hitIndex].x = xx;
    fHitData[row->fHitNumberOffset + hitIndex].y = yy;
  }
}

void AliHLTTPCCASliceData::Clear()
{
  fNumberOfHits = 0;
}

void AliHLTTPCCASliceData::InitializeRows( const AliHLTTPCCAParam &p )
{
  // initialisation of rows
	if (!fRows) fRows = new AliHLTTPCCARow[HLTCA_ROW_COUNT + 1];
  for ( int i = 0; i < p.NRows(); ++i ) {
    fRows[i].fX = p.RowX( i );
    fRows[i].fMaxY = CAMath::Tan( p.DAlpha() / 2. ) * fRows[i].fX;
  }
}

#ifndef HLTCA_GPUCODE
	AliHLTTPCCASliceData::~AliHLTTPCCASliceData()
	{
		//Standard Destrcutor
		if (fRows)
		{
			if (!fIsGpuSliceData) delete[] fRows;
			fRows = NULL;
		}
		if (fMemory)
		{
			if (!fIsGpuSliceData) delete[] fMemory;
			fMemory = NULL;
		}

	}
#endif

GPUh() void AliHLTTPCCASliceData::SetGPUSliceDataMemory(void* const pSliceMemory, void* const pRowMemory)
{
	//Set Pointer to slice data memory to external memory
	fMemory = (char*) pSliceMemory;
	fRows = (AliHLTTPCCARow*) pRowMemory;
}

size_t AliHLTTPCCASliceData::SetPointers(const AliHLTTPCCAClusterData *data, bool allocate)
{
	//Set slice data internal pointers
  int hitMemCount = 0;
  for ( int rowIndex = data->FirstRow(); rowIndex <= data->LastRow(); ++rowIndex )
  {
	hitMemCount += NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(data->NumberOfClusters( rowIndex ));
  }
	//Calculate Memory needed to store hits in rows

  const int numberOfRows = data->LastRow() - data->FirstRow() + 1;
  const unsigned int kVectorAlignment = 256 /*sizeof( uint4 )*/ ;
  fNumberOfHitsPlusAlign = NextMultipleOf < ( kVectorAlignment > sizeof(HLTCA_GPU_ROWALIGNMENT) ? kVectorAlignment : sizeof(HLTCA_GPU_ROWALIGNMENT)) / sizeof( int ) > ( hitMemCount );
  fNumberOfHits = data->NumberOfClusters();
  const int firstHitInBinSize = (23 + sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(int)) * numberOfRows + 4 * fNumberOfHits + 3;
  //FIXME: sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(int) * numberOfRows is way to big and only to ensure to reserve enough memory for GPU Alignment.
  //Might be replaced by correct value

  const int memorySize =
    // LinkData, HitData
    fNumberOfHitsPlusAlign * 4 * sizeof( short ) +
    // FirstHitInBin
    NextMultipleOf<kVectorAlignment>( ( firstHitInBinSize ) * sizeof( int ) ) +
    // HitWeights, ClusterDataIndex
    fNumberOfHitsPlusAlign * 2 * sizeof( int );

  if ( fMemorySize < memorySize ) {
	fMemorySize = memorySize;
	if (allocate && !fIsGpuSliceData)
	{
		if (fMemory)
		{
			delete[] fMemory;
		}
	  fMemory = new char[fMemorySize + 4];// kVectorAlignment];
	}
  }

  char *mem = fMemory;
  AssignMemory( fLinkUpData,   mem, fNumberOfHitsPlusAlign );
  AssignMemory( fLinkDownData, mem, fNumberOfHitsPlusAlign );
  AssignMemory( fHitData,     mem, fNumberOfHitsPlusAlign );
  AssignMemory( fFirstHitInBin,  mem, firstHitInBinSize );
  fGpuMemorySize = mem - fMemory;

  //Memory Allocated below will not be copied to GPU but instead be initialized on the gpu itself. Therefore it must not be copied to GPU!
  AssignMemory( fHitWeights,   mem, fNumberOfHitsPlusAlign );
  AssignMemory( fClusterDataIndex, mem, fNumberOfHitsPlusAlign );
  return(mem - fMemory);
}

void AliHLTTPCCASliceData::InitFromClusterData( const AliHLTTPCCAClusterData &data )
{
  // initialisation from cluster data

  ////////////////////////////////////
  // 1. prepare arrays
  ////////////////////////////////////

  const int numberOfRows = data.LastRow() - data.FirstRow() + 1;
  fNumberOfHits = data.NumberOfClusters();

  /* TODO Vectorization
  for ( int rowIndex = data.FirstRow(); rowIndex <= data.LastRow(); ++rowIndex ) {
    int NumberOfClusters( int rowIndex ) const;
  }
  const int memorySize = fNumberOfHits * sizeof( short_v::Type )
  */
  SetPointers(&data, true);

  ////////////////////////////////////
  // 2. fill HitData and FirstHitInBin
  ////////////////////////////////////

  for ( int rowIndex = 0; rowIndex < data.FirstRow(); ++rowIndex ) {
    AliHLTTPCCARow &row = fRows[rowIndex];
    row.fGrid.CreateEmpty();
    row.fNHits = 0;
    row.fFullSize = 0;
    row.fHitNumberOffset = 0;
    row.fFirstHitInBinOffset = 0;

    row.fHy0 = 0.f;
    row.fHz0 = 0.f;
    row.fHstepY = 1.f;
    row.fHstepZ = 1.f;
    row.fHstepYi = 1.f;
    row.fHstepZi = 1.f;
  }
  for ( int rowIndex = data.LastRow() + 1; rowIndex < HLTCA_ROW_COUNT + 1; ++rowIndex ) {
    AliHLTTPCCARow &row = fRows[rowIndex];
    row.fGrid.CreateEmpty();
    row.fNHits = 0;
    row.fFullSize = 0;
    row.fHitNumberOffset = 0;
    row.fFirstHitInBinOffset = 0;

    row.fHy0 = 0.f;
    row.fHz0 = 0.f;
    row.fHstepY = 1.f;
    row.fHstepZ = 1.f;
    row.fHstepYi = 1.f;
    row.fHstepZi = 1.f;
  }


  AliHLTResizableArray<AliHLTTPCCAHit> binSortedHits( fNumberOfHits + sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v) * numberOfRows + 1 );

  int gridContentOffset = 0;
  int hitOffset = 0;

  int binCreationMemorySize = 103 * 2 + fNumberOfHits;
  AliHLTResizableArray<unsigned short> binCreationMemory( binCreationMemorySize );

  fGPUSharedDataReq = 0;

  for ( int rowIndex = data.FirstRow(); rowIndex <= data.LastRow(); ++rowIndex ) {
    AliHLTTPCCARow &row = fRows[rowIndex];
    row.fNHits = data.NumberOfClusters( rowIndex );
    assert( row.fNHits < ( 1 << sizeof( unsigned short ) * 8 ) );
	row.fHitNumberOffset = hitOffset;
	hitOffset += NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(data.NumberOfClusters( rowIndex ));

    row.fFirstHitInBinOffset = gridContentOffset;

    CreateGrid( &row, data, data.RowOffset( rowIndex ) );
    const AliHLTTPCCAGrid &grid = row.fGrid;
    const int numberOfBins = grid.N();

    int binCreationMemorySizeNew;
    if ( ( binCreationMemorySizeNew = numberOfBins * 2 + 6 + row.fNHits + sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(unsigned short) * numberOfRows + 1) > binCreationMemorySize ) {
      binCreationMemorySize = binCreationMemorySizeNew;
      binCreationMemory.Resize( binCreationMemorySize );
    }

    AliHLTArray<unsigned short> c = binCreationMemory;           // number of hits in all previous bins
    AliHLTArray<unsigned short> bins = c + ( numberOfBins + 3 ); // cache for the bin index for every hit in this row
    AliHLTArray<unsigned short> filled = bins + row.fNHits;      // counts how many hits there are per bin

    for ( unsigned int bin = 0; bin < row.fGrid.N() + 3; ++bin ) {
      filled[bin] = 0; // initialize filled[] to 0
    }

    for ( int hitIndex = 0; hitIndex < row.fNHits; ++hitIndex ) {
      const int globalHitIndex = data.RowOffset( rowIndex ) + hitIndex;
      const unsigned short bin = row.fGrid.GetBin( data.Y( globalHitIndex ), data.Z( globalHitIndex ) );

      bins[hitIndex] = bin;
      ++filled[bin];
    }

    unsigned short n = 0;
    for ( int bin = 0; bin < numberOfBins + 3; ++bin ) {
      c[bin] = n;
      n += filled[bin];
    }

    for ( int hitIndex = 0; hitIndex < row.fNHits; ++hitIndex ) {
      const unsigned short bin = bins[hitIndex];
      --filled[bin];
      const unsigned short ind = c[bin] + filled[bin]; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
      const int globalBinsortedIndex = row.fHitNumberOffset + ind;
      const int globalHitIndex = data.RowOffset( rowIndex ) + hitIndex;

      // allows to find the global hit index / coordinates from a global bin sorted hit index
      fClusterDataIndex[globalBinsortedIndex] = globalHitIndex;
      binSortedHits[globalBinsortedIndex].SetY( data.Y( globalHitIndex ) );
      binSortedHits[globalBinsortedIndex].SetZ( data.Z( globalHitIndex ) );
    }

    PackHitData( &row, binSortedHits );

    for ( int i = 0; i < numberOfBins; ++i ) {
      fFirstHitInBin[row.fFirstHitInBinOffset + i] = c[i]; // global bin-sorted hit index
    }
    const unsigned short a = c[numberOfBins];
    // grid.N is <= row.fNHits
    const int nn = numberOfBins + grid.Ny() + 3;
    for ( int i = numberOfBins; i < nn; ++i ) {
      assert( (signed) row.fFirstHitInBinOffset + i < 23 * numberOfRows + 4 * fNumberOfHits + 3 );
      fFirstHitInBin[row.fFirstHitInBinOffset + i] = a;
    }

    row.fFullSize = nn;
    gridContentOffset += nn;

	if (NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.fNHits) + nn > (unsigned) fGPUSharedDataReq)
		fGPUSharedDataReq = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(row.fNHits) + nn;

	//Make pointer aligned
	gridContentOffset = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(gridContentOffset);
  }

#if 0
  //SG cell finder - test code

  if ( fTmpHitInputIDs ) delete[] fTmpHitInputIDs;
  fTmpHitInputIDs = new int [NHits];
  const float areaY = .5;
  const float areaZ = .5;
  int newRowNHitsTotal = 0;
  bool *usedHits = new bool [NHits];
  for ( int iHit = 0; iHit < NHits; iHit++ ) usedHits[iHit] = 0;
  for ( int iRow = 0; iRow < fParam.NRows(); iRow++ ) {
    rowHeaders[iRow*2  ] = newRowNHitsTotal; // new first hit
    rowHeaders[iRow*2+1] = 0; // new N hits
    int newRowNHits = 0;
    int oldRowFirstHit = RowFirstHit[iRow];
    int oldRowLastHit = oldRowFirstHit + RowNHits[iRow];
    for ( int iHit = oldRowFirstHit; iHit < oldRowLastHit; iHit++ ) {
      if ( usedHits[iHit] ) continue;
      float x0 = X[iHit];
      float y0 = Y[iHit];
      float z0 = Z[iHit];
      float cx = x0;
      float cy = y0;
      float cz = z0;
      int nclu = 1;
      usedHits[iHit] = 1;
      if ( 0 ) for ( int jHit = iHit + 1; jHit < oldRowLastHit; jHit++ ) {//SG!!!
          //if( usedHits[jHit] ) continue;
          float dy = Y[jHit] - y0;
          float dz = Z[jHit] - z0;
          if ( CAMath::Abs( dy ) < areaY && CAMath::Abs( dz ) < areaZ ) {
            cx += X[jHit];
            cy += Y[jHit];
            cz += Z[jHit];
            nclu++;
            usedHits[jHit] = 1;
          }
        }
      int id = newRowNHitsTotal + newRowNHits;
      hitsXYZ[id*3+0 ] = cx / nclu;
      hitsXYZ[id*3+1 ] = cy / nclu;
      hitsXYZ[id*3+2 ] = cz / nclu;
      fTmpHitInputIDs[id] = iHit;
      newRowNHits++;
    }
    rowHeaders[iRow*2+1] = newRowNHits;
    newRowNHitsTotal += newRowNHits;
  }
  NHitsTotal() = newRowNHitsTotal;
  reinterpret_cast<int*>( fInputEvent )[1+fParam.NRows()*2] = newRowNHitsTotal;

  delete[] usedHits;
#endif
}

void AliHLTTPCCASliceData::ClearHitWeights()
{
  // clear hit weights

#ifdef ENABLE_VECTORIZATION
  const int_v v0( Zero );
  const int *const end = fHitWeights + fNumberOfHits;
  for ( int *mem = fHitWeights; mem < end; mem += v0.Size ) {
    v0.store( mem );
  }
#else
  for ( int i = 0; i < fNumberOfHitsPlusAlign; ++i ) {
    fHitWeights[i] = 0;
  }
#endif
}

void AliHLTTPCCASliceData::ClearLinks()
{
  // link cleaning

#ifdef ENABLE_VECTORIZATION
  const short_v v0( -1 );
  const short *const end1 = fLinkUpData + fNumberOfHits;
  for ( short *mem = fLinkUpData; mem < end; mem += v0.Size ) {
    v0.store( mem );
  }
  const short *const end2 = fLinkDownData + fNumberOfHits;
  for ( short *mem = fLinkDownData; mem < end; mem += v0.Size ) {
    v0.store( mem );
  }
#else
  for ( int i = 0; i < fNumberOfHits; ++i ) {
    fLinkUpData[i] = -1;
  }
  for ( int i = 0; i < fNumberOfHits; ++i ) {
    fLinkDownData[i] = -1;
  }
#endif
}

