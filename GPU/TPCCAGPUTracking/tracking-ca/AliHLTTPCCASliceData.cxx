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
#include "AliHLTTPCCAGPUConfig.h"
#include "AliHLTTPCCAGPUTracker.h"
#include "MemoryAssignmentHelpers.h"
#include <iostream>
#include <string.h>

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

inline void AliHLTTPCCASliceData::CreateGrid( AliHLTTPCCARow *row, const float2* data, int ClusterDataHitNumberOffset )
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
    const float y = data[i].x;
    const float z = data[i].y;
    if ( yMax < y ) yMax = y;
    if ( yMin > y ) yMin = y;
    if ( zMax < z ) zMax = z;
    if ( zMin > z ) zMin = z;
  }

  float dz = zMax - zMin;
  float tfFactor = 1.;
  if (dz > 270.)
  {
      tfFactor = dz / 250.;
      dz = 250.;
  }
  const float norm = fastInvSqrt( row->fNHits / tfFactor );
  row->fGrid.Create( yMin, yMax, zMin, zMax,
                     CAMath::Max( ( yMax - yMin ) * norm, 2.f ),
                     CAMath::Max( dz * norm, 2.f ) );
}

inline int AliHLTTPCCASliceData::PackHitData( AliHLTTPCCARow* const row, const AliHLTArray<AliHLTTPCCAHit> &binSortedHits )
{
  // hit data packing

  static const float maxVal = (((long long int) 1 << AliHLTTPCCAMath::Min(24, sizeof(cahit) * 8)) - 1); //Stay within float precision in any case!
  static const float packingConstant = 1.f / (maxVal - 2.);
  const float y0 = row->fGrid.YMin();
  const float z0 = row->fGrid.ZMin();
  const float stepY = ( row->fGrid.YMax() - y0 ) * packingConstant;
  const float stepZ = ( row->fGrid.ZMax() - z0 ) * packingConstant;
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
    const AliHLTTPCCAHit &hh = binSortedHits[hitIndex];
    const float xx = ( ( hh.Y() - y0 ) * stepYi ) + .5 ;
    const float yy = ( ( hh.Z() - z0 ) * stepZi ) + .5 ;
    if ( xx < 0 || yy < 0 || xx > maxVal  || yy > maxVal ) {
      std::cout << "!!!! hit packing error!!! " << xx << " " << yy << " (" << maxVal << ")" << std::endl;
      return 1;
    }
    // HitData is bin sorted
    fHitData[globalHitIndex].x = (cahit) xx;
    fHitData[globalHitIndex].y = (cahit) yy;
  }
  return 0;
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

  int hitMemCount = HLTCA_ROW_COUNT * sizeof(HLTCA_GPU_ROWALIGNMENT) + data->NumberOfClusters();
  //Calculate Memory needed to store hits in rows

  const unsigned int kVectorAlignment = 256 /*sizeof( uint4 )*/ ;
  fNumberOfHitsPlusAlign = NextMultipleOf < ( kVectorAlignment > sizeof(HLTCA_GPU_ROWALIGNMENT) ? kVectorAlignment : sizeof(HLTCA_GPU_ROWALIGNMENT)) / sizeof( int ) > ( hitMemCount );
  fNumberOfHits = data->NumberOfClusters();
  const int firstHitInBinSize = (23 + sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(int)) * HLTCA_ROW_COUNT + 4 * fNumberOfHits + 3;
  //FIXME: sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(int) * HLTCA_ROW_COUNT is way to big and only to ensure to reserve enough memory for GPU Alignment.
  //Might be replaced by correct value

  const int memorySize =
    // LinkData, HitData
    fNumberOfHitsPlusAlign * 2 * (sizeof( cahit ) + sizeof( calink )) +
    // FirstHitInBin
    NextMultipleOf<kVectorAlignment>( ( firstHitInBinSize ) * sizeof( int ) ) +
    // HitWeights, ClusterDataIndex
    fNumberOfHitsPlusAlign * 2 * sizeof( int );

  fMemorySize = memorySize + 4;
  if (allocate)
  {
  	if (!fIsGpuSliceData)
  	{
  		if (fMemory)
  		{
  			delete[] fMemory;
  		}
  		fMemory = new char[fMemorySize];// kVectorAlignment];
  	}
  	else
  	{
  		if (fMemorySize > HLTCA_GPU_SLICE_DATA_MEMORY)
  		{
  			return(0);
  		}
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

int AliHLTTPCCASliceData::InitFromClusterData( const AliHLTTPCCAClusterData &data )
{
  // initialisation from cluster data

  ////////////////////////////////////
  // 0. sort rows
  ////////////////////////////////////

  fNumberOfHits = data.NumberOfClusters();
  fMaxZ = 0.f;

  float2* YZData = new float2[fNumberOfHits];
  int* tmpHitIndex = new int[fNumberOfHits];

  int RowOffset[HLTCA_ROW_COUNT];
  int NumberOfClustersInRow[HLTCA_ROW_COUNT];
  memset(NumberOfClustersInRow, 0, HLTCA_ROW_COUNT * sizeof(int));
  fFirstRow = HLTCA_ROW_COUNT;
  fLastRow = 0;

  for (int i = 0;i < fNumberOfHits;i++)
  {
    const int tmpRow = data.RowNumber(i);
	NumberOfClustersInRow[tmpRow]++;
	if (tmpRow > fLastRow) fLastRow = tmpRow;
	if (tmpRow < fFirstRow) fFirstRow = tmpRow;
  }
  int tmpOffset = 0;
  for (int i = fFirstRow;i <= fLastRow;i++)
  {
      if ((long long int) NumberOfClustersInRow[i] >= ((long long int) 1 << (sizeof(calink) * 8)))
      {
        printf("Too many clusters in row %d for row indexing (%d >= %lld), indexing insufficient\n", i, NumberOfClustersInRow[i], ((long long int) 1 << (sizeof(calink) * 8)));
        return(1);
      }
      if (NumberOfClustersInRow[i] >= (1 << 24))
      {
        printf("Too many clusters in row %d for hit id indexing (%d >= %d), indexing insufficient\n", i, NumberOfClustersInRow[i], 1 << 24);
        return(1);
      }
	  RowOffset[i] = tmpOffset;
	  tmpOffset += NumberOfClustersInRow[i];
  }
  
  {
	  int RowsFilled[HLTCA_ROW_COUNT];
	  memset(RowsFilled, 0, HLTCA_ROW_COUNT * sizeof(int));
	  for (int i = 0;i < fNumberOfHits;i++)
	  {
		float2 tmp;
		tmp.x = data.Y(i);
		tmp.y = data.Z(i);
        if (fabs(tmp.y) > fMaxZ) fMaxZ = fabs(tmp.y);
		int tmpRow = data.RowNumber(i);
		int newIndex = RowOffset[tmpRow] + (RowsFilled[tmpRow])++;
		YZData[newIndex] = tmp;
		tmpHitIndex[newIndex] = i;
	  }
  }
  if (fFirstRow == HLTCA_ROW_COUNT) fFirstRow = 0;

  ////////////////////////////////////
  // 1. prepare arrays
  ////////////////////////////////////

  const int numberOfRows = fLastRow - fFirstRow + 1;

  if (SetPointers(&data, true) == 0)
  {
	delete[] YZData;
	delete[] tmpHitIndex;
	return 1;
  }

  ////////////////////////////////////
  // 2. fill HitData and FirstHitInBin
  ////////////////////////////////////

  for ( int rowIndex = 0; rowIndex < fFirstRow; ++rowIndex ) {
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
  for ( int rowIndex = fLastRow + 1; rowIndex < HLTCA_ROW_COUNT + 1; ++rowIndex ) {
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


  AliHLTResizableArray<AliHLTTPCCAHit> binSortedHits( fNumberOfHits + sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v));

  int gridContentOffset = 0;
  int hitOffset = 0;

  int binCreationMemorySize = 103 * 2 + fNumberOfHits;
  AliHLTResizableArray<calink> binCreationMemory( binCreationMemorySize );

  fGPUSharedDataReq = 0;

  for ( int rowIndex = fFirstRow; rowIndex <= fLastRow; ++rowIndex ) {
    AliHLTTPCCARow &row = fRows[rowIndex];
	row.fNHits = NumberOfClustersInRow[rowIndex];
	row.fHitNumberOffset = hitOffset;
	hitOffset += NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(ushort_v)>(NumberOfClustersInRow[rowIndex]);

    row.fFirstHitInBinOffset = gridContentOffset;

    CreateGrid( &row, YZData, RowOffset[rowIndex] );
    const AliHLTTPCCAGrid &grid = row.fGrid;
    const int numberOfBins = grid.N();
    if ((long long int) numberOfBins >= ((long long int) 1 << (sizeof(calink) * 8)))
    {
      printf("Too many bins in row %d for grid (%d >= %lld), indexing insufficient\n", rowIndex, numberOfBins, ((long long int) 1 << (sizeof(calink) * 8)));
      delete[] YZData;
      delete[] tmpHitIndex;
      return(1);
    }

    int binCreationMemorySizeNew;
    if ( ( binCreationMemorySizeNew = numberOfBins * 2 + 6 + row.fNHits + sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(unsigned short) * numberOfRows + 1) > binCreationMemorySize ) {
      binCreationMemorySize = binCreationMemorySizeNew;
      binCreationMemory.Resize( binCreationMemorySize );
    }

    AliHLTArray<calink> c = binCreationMemory;           // number of hits in all previous bins
    AliHLTArray<calink> bins = c + ( numberOfBins + 3 ); // cache for the bin index for every hit in this row, 3 extra empty bins at the end!!!
    AliHLTArray<calink> filled = bins + row.fNHits;      // counts how many hits there are per bin

    for ( unsigned int bin = 0; bin < row.fGrid.N() + 3; ++bin ) {
      filled[bin] = 0; // initialize filled[] to 0
    }

    for ( int hitIndex = 0; hitIndex < row.fNHits; ++hitIndex ) {
      const int globalHitIndex = RowOffset[rowIndex] + hitIndex;
      const calink bin = row.fGrid.GetBin( YZData[globalHitIndex].x, YZData[globalHitIndex].y );

      bins[hitIndex] = bin;
      ++filled[bin];
    }

    calink n = 0;
    for ( int bin = 0; bin < numberOfBins + 3; ++bin ) {
      c[bin] = n;
      n += filled[bin];
    }

    for ( int hitIndex = 0; hitIndex < row.fNHits; ++hitIndex ) {
      const calink bin = bins[hitIndex];
      --filled[bin];
      const calink ind = c[bin] + filled[bin]; // generate an index for this hit that is >= c[bin] and < c[bin + 1]
      const int globalBinsortedIndex = row.fHitNumberOffset + ind;
      const int globalHitIndex = RowOffset[rowIndex] + hitIndex;

      // allows to find the global hit index / coordinates from a global bin sorted hit index
      fClusterDataIndex[globalBinsortedIndex] = tmpHitIndex[globalHitIndex];
      binSortedHits[ind].SetY( YZData[globalHitIndex].x );
      binSortedHits[ind].SetZ( YZData[globalHitIndex].y );
    }

    if (PackHitData( &row, binSortedHits ))
    {
      delete[] YZData;
      delete[] tmpHitIndex;
      return(1);
    }

    for ( int i = 0; i < numberOfBins; ++i ) {
      fFirstHitInBin[row.fFirstHitInBinOffset + i] = c[i]; // global bin-sorted hit index
    }
    const calink a = c[numberOfBins];
    // grid.N is <= row.fNHits
    const int nn = numberOfBins + grid.Ny() + 3;
    for ( int i = numberOfBins; i < nn; ++i ) {
      assert( (signed) row.fFirstHitInBinOffset + i < 23 * numberOfRows + 4 * fNumberOfHits + 3 );
      fFirstHitInBin[row.fFirstHitInBinOffset + i] = a;
    }

    row.fFullSize = nn;
    gridContentOffset += nn;

	if (NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(calink)>(row.fNHits) + nn > (unsigned) fGPUSharedDataReq)
		fGPUSharedDataReq = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(calink)>(row.fNHits) + nn;

	//Make pointer aligned
	gridContentOffset = NextMultipleOf<sizeof(HLTCA_GPU_ROWALIGNMENT) / sizeof(calink)>(gridContentOffset);
  }

  delete[] YZData;
  delete[] tmpHitIndex;

  return(0);
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

  for ( int i = 0; i < fNumberOfHits; ++i ) {
    fLinkUpData[i] = CALINK_INVAL;
  }
  for ( int i = 0; i < fNumberOfHits; ++i ) {
    fLinkDownData[i] = CALINK_INVAL;
  }
}
