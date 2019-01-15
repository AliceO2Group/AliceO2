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

#ifndef ALIHLTTPCCASLICEDATA_H
#define ALIHLTTPCCASLICEDATA_H

#include "AliGPUTPCDef.h"
#include "AliGPUTPCRow.h"
#include "AliTPCCommonMath.h"
#include "AliGPUCAParam.h"
#include "AliGPUProcessor.h"
#include "AliGPUTPCGPUConfig.h"

class AliGPUTPCClusterData;
class AliGPUTPCHit;

/**
 * Data abstraction class for the Slice Tracker.
 *
 * Different architectures implement this for the most efficient loads and stores. All access to the
 * data happens through inline functions so that access to the data has no extra costs.
 */
MEM_CLASS_PRE() class AliGPUTPCSliceData : public AliGPUProcessor
{
  public:
	AliGPUTPCSliceData() :
        AliGPUProcessor(),
		fFirstRow( 0 ), fLastRow( GPUCA_ROW_COUNT - 1), fNumberOfHits( 0 ), fNumberOfHitsPlusAlign( 0 ), fMaxZ(0.f), fMemorySize( 0 ), fGpuMemorySize( 0 ), fMemory( 0 ), fGPUTextureBase( 0 )
		,fRows( NULL ), fLinkUpData( 0 ), fLinkDownData( 0 ), fHitData( 0 ), fClusterDataIndex( 0 )
		, fFirstHitInBin( 0 ), fHitWeights( 0 )
	{
	}

#ifndef GPUCA_GPUCODE
	~AliGPUTPCSliceData();
#endif //!GPUCA_GPUCODE

	MEM_CLASS_PRE2() void InitializeRows( const MEM_LG2(AliGPUCAParam) &parameters );

	/**
	 * (Re)Create the data that is tuned for optimal performance of the algorithm from the cluster
	 * data.
	 */

	void SetGPUSliceDataMemory(void* const pSliceMemory, void* const pRowMemory);
	size_t SetPointers(const AliGPUTPCClusterData *data, bool allocate = false);
	int InitFromClusterData( const AliGPUTPCClusterData &data );

	/**
	 * Clear the slice data (e.g. for an empty slice)
	 */
	void Clear();

	/**
	 * Return the number of hits in this slice.
	 */
	GPUhd() int NumberOfHits() const { return fNumberOfHits; }
	GPUhd() int NumberOfHitsPlusAlign() const { return fNumberOfHitsPlusAlign; }

	/**
	 * Access to the hit links.
	 *
	 * The links values give the hit index in the row above/below. Or -1 if there is no link.
	 */
	MEM_TEMPLATE() GPUd() calink HitLinkUpData  ( const MEM_TYPE(AliGPUTPCRow) &row, const calink &hitIndex ) const;
	MEM_TEMPLATE() GPUd() calink HitLinkDownData( const MEM_TYPE(AliGPUTPCRow) &row, const calink &hitIndex ) const;

	MEM_TEMPLATE() GPUhdi() GPUglobalref() const cahit2 *HitData( const MEM_TYPE(AliGPUTPCRow) &row ) const {return &fHitData[row.fHitNumberOffset];}
	GPUhd() GPUglobalref() const cahit2* HitData() const { return(fHitData); }
	MEM_TEMPLATE() GPUdi() GPUglobalref() const calink *HitLinkUpData  ( const MEM_TYPE(AliGPUTPCRow) &row ) const {return &fLinkUpData[row.fHitNumberOffset];}
	MEM_TEMPLATE() GPUdi() GPUglobalref() const calink *HitLinkDownData( const MEM_TYPE(AliGPUTPCRow) &row ) const {return &fLinkDownData[row.fHitNumberOffset];}
	MEM_TEMPLATE() GPUdi() GPUglobalref() const calink *FirstHitInBin( const MEM_TYPE( AliGPUTPCRow) &row ) const {return &fFirstHitInBin[row.fFirstHitInBinOffset];}

	MEM_TEMPLATE() GPUd() void SetHitLinkUpData  ( const MEM_TYPE(AliGPUTPCRow) &row, const calink &hitIndex, const calink &value );
	MEM_TEMPLATE() GPUd() void SetHitLinkDownData( const MEM_TYPE(AliGPUTPCRow) &row, const calink &hitIndex, const calink &value );
	/**
	 * Reset all links to -1.
	 */
	void ClearLinks();

	/**
	 * Return the y and z coordinate(s) of the given hit(s).
	 */
	MEM_TEMPLATE() GPUd() cahit HitDataY( const MEM_TYPE( AliGPUTPCRow) &row, const unsigned int &hitIndex ) const;
	MEM_TEMPLATE() GPUd() cahit HitDataZ( const MEM_TYPE( AliGPUTPCRow) &row, const unsigned int &hitIndex ) const;
	MEM_TEMPLATE() GPUd() cahit2 HitData( const MEM_TYPE( AliGPUTPCRow) &row, const unsigned int &hitIndex ) const;

	/**
	 * For a given bin index, content tells how many hits there are in the preceding bins. This maps
	 * directly to the hit index in the given row.
	 *
	 * \param binIndexes in the range 0 to row.Grid.N + row.Grid.Ny + 3.
	 */
	MEM_TEMPLATE() GPUd() calink FirstHitInBin( const MEM_TYPE( AliGPUTPCRow)&row, calink binIndexes ) const;

	/**
	 * If the given weight is higher than what is currently stored replace with the new weight.
	 */
	MEM_TEMPLATE() GPUd() void MaximizeHitWeight( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex, int weight );
	MEM_TEMPLATE() GPUd() void SetHitWeight( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex, int weight );

	/**
	 * Return the maximal weight the given hit got from one tracklet
	 */
	MEM_TEMPLATE() GPUd() int HitWeight( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex ) const;

	/**
	 * Reset all hit weights to 0.
	 */
	void ClearHitWeights();

	/**
	 * Returns the index in the original AliGPUTPCClusterData object of the given hit
	 */
	MEM_TEMPLATE() GPUhd() int ClusterDataIndex( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex ) const;

	/**
	 * Return the row object for the given row index.
	 */
	GPUhdi() GPUglobalref() const MEM_GLOBAL(AliGPUTPCRow)& Row( int rowIndex ) const {return fRows[rowIndex];}
	GPUhdi() GPUglobalref() MEM_GLOBAL(AliGPUTPCRow)* Rows() const {return fRows;}

	GPUhdi() GPUglobalref() int* HitWeights() const {return(fHitWeights); }

	GPUhdi() void SetGPUTextureBase(char* const val) {fGPUTextureBase = val;}
	GPUhdi() char* GPUTextureBase() const { return(fGPUTextureBase); }
	GPUhdi() char* GPUTextureBaseConst() const { return(fGPUTextureBase); }

#if !defined(__OPENCL__)
	GPUhi() char* Memory() const {return(fMemory); }
	GPUhi() size_t MemorySize() const {return(fMemorySize); }
	GPUhi() size_t GpuMemorySize() const {return(fGpuMemorySize); }
#endif

	float MaxZ() const { return fMaxZ; }

  private:
	AliGPUTPCSliceData( const AliGPUTPCSliceData & );
	AliGPUTPCSliceData& operator=( const AliGPUTPCSliceData & ) ;

#ifndef GPUCA_GPUCODE
	void CreateGrid( AliGPUTPCRow *row, const float2* data, int ClusterDataHitNumberOffset );
	int PackHitData( AliGPUTPCRow *row, const AliGPUTPCHit* binSortedHits );
#endif

	int fFirstRow;             //First non-empty row
	int fLastRow;              //Last non-empty row

	int fNumberOfHits;         // the number of hits in this slice
	int fNumberOfHitsPlusAlign;
    
	float fMaxZ;

	size_t fMemorySize;           // size of the allocated memory in bytes
	size_t fGpuMemorySize;        // size of Memory needed to be transfered to GPU
	GPUglobalref() char *fMemory;             // pointer to the allocated memory where all the following arrays reside in
	GPUglobalref() char *fGPUTextureBase;     // pointer to start of GPU texture

	GPUglobalref() MEM_GLOBAL(AliGPUTPCRow) *fRows;     // The row objects needed for most accessor functions

	GPUglobalref() calink *fLinkUpData;        // hit index in the row above which is linked to the given (global) hit index
	GPUglobalref() calink *fLinkDownData;      // hit index in the row below which is linked to the given (global) hit index

	GPUglobalref() cahit2 *fHitData;         // packed y,z coordinate of the given (global) hit index

	GPUglobalref() int *fClusterDataIndex;    // see ClusterDataIndex()

	/*
	 * The size of the array is row.Grid.N + row.Grid.Ny + 3. The row.Grid.Ny + 3 is an optimization
	 * to remove the need for bounds checking. The last values are the same as the entry at [N - 1].
	 */
	GPUglobalref() calink *fFirstHitInBin;         // see FirstHitInBin

	GPUglobalref() int *fHitWeights;          // the weight of the longest tracklet crossed the cluster

};

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() calink MEM_LG(AliGPUTPCSliceData)::HitLinkUpData  ( const MEM_TYPE( AliGPUTPCRow)&row, const calink &hitIndex ) const
{
	return fLinkUpData[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() calink MEM_LG(AliGPUTPCSliceData)::HitLinkDownData( const MEM_TYPE( AliGPUTPCRow)&row, const calink &hitIndex ) const
{
	return fLinkDownData[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() void MEM_LG(AliGPUTPCSliceData)::SetHitLinkUpData  ( const MEM_TYPE( AliGPUTPCRow)&row, const calink &hitIndex, const calink &value )
{
	fLinkUpData[row.fHitNumberOffset + hitIndex] = value;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() void MEM_LG(AliGPUTPCSliceData)::SetHitLinkDownData( const MEM_TYPE( AliGPUTPCRow)&row, const calink &hitIndex, const calink &value )
{
	fLinkDownData[row.fHitNumberOffset + hitIndex] = value;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() cahit MEM_LG(AliGPUTPCSliceData)::HitDataY( const MEM_TYPE( AliGPUTPCRow)&row, const unsigned int &hitIndex ) const
{
	return fHitData[row.fHitNumberOffset + hitIndex].x;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() cahit MEM_LG(AliGPUTPCSliceData)::HitDataZ( const MEM_TYPE( AliGPUTPCRow)&row, const unsigned int &hitIndex ) const
{
	return fHitData[row.fHitNumberOffset + hitIndex].y;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() cahit2 MEM_LG(AliGPUTPCSliceData)::HitData( const MEM_TYPE( AliGPUTPCRow)&row, const unsigned int &hitIndex ) const
{
	return fHitData[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() calink MEM_LG(AliGPUTPCSliceData)::FirstHitInBin( const MEM_TYPE( AliGPUTPCRow)&row, calink binIndexes ) const
{
	return fFirstHitInBin[row.fFirstHitInBinOffset + binIndexes];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUhdi() int MEM_LG(AliGPUTPCSliceData)::ClusterDataIndex( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex ) const
{
	return fClusterDataIndex[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() void MEM_LG(AliGPUTPCSliceData)::MaximizeHitWeight( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex, int weight )
{
	CAMath::AtomicMax( &fHitWeights[row.fHitNumberOffset + hitIndex], weight );
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() void MEM_LG(AliGPUTPCSliceData)::SetHitWeight( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex, int weight )
{
	fHitWeights[row.fHitNumberOffset + hitIndex] = weight;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUdi() int MEM_LG(AliGPUTPCSliceData)::HitWeight( const MEM_TYPE( AliGPUTPCRow)&row, unsigned int hitIndex ) const
{
	return fHitWeights[row.fHitNumberOffset + hitIndex];
}

#endif // ALIHLTTPCCASLICEDATA_H
