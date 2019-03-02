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

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCAMath.h"
#if !(defined(HLTCA_GPUCODE) && defined(__OPENCL__) && !defined(HLTCA_HOSTCODE))
#include "AliHLTArray.h"
#endif
#include "AliHLTTPCCAGPUConfig.h"

typedef int int_v;
typedef unsigned int uint_v;
typedef short short_v;
typedef unsigned short ushort_v;
typedef float float_v;

class AliHLTTPCCAClusterData;
#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
template<typename T, int Dim> class AliHLTArray;
#endif
class AliHLTTPCCAHit;
MEM_CLASS_PRE() class AliHLTTPCCAParam;

/**
 * Data abstraction class for the Slice Tracker.
 *
 * Different architectures implement this for the most efficient loads and stores. All access to the
 * data happens through inline functions so that access to the data has no extra costs.
 */
MEM_CLASS_PRE() class AliHLTTPCCASliceData
{
  public:
    AliHLTTPCCASliceData()
      : 
      fIsGpuSliceData(0), fGPUSharedDataReq(0), fFirstRow( 0 ), fLastRow( HLTCA_ROW_COUNT - 1), fNumberOfHits( 0 ), fNumberOfHitsPlusAlign( 0 ), fMaxZ(0.f), fMemorySize( 0 ), fGpuMemorySize( 0 ), fMemory( 0 ), fGPUTextureBase( 0 )
      ,fRows( NULL ), fLinkUpData( 0 ), fLinkDownData( 0 ), fHitData( 0 ), fClusterDataIndex( 0 )
      , fFirstHitInBin( 0 ), fHitWeights( 0 )
    {
    }

#ifndef HLTCA_GPUCODE
    ~AliHLTTPCCASliceData();
#endif //!HLTCA_GPUCODE

    MEM_CLASS_PRE2() void InitializeRows( const MEM_LG2(AliHLTTPCCAParam) &parameters );

    /**
     * (Re)Create the data that is tuned for optimal performance of the algorithm from the cluster
     * data.
     */

    void SetGPUSliceDataMemory(void* const pSliceMemory, void* const pRowMemory);
    size_t SetPointers(const AliHLTTPCCAClusterData *data, bool allocate = false);
    int InitFromClusterData( const AliHLTTPCCAClusterData &data );

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
    MEM_TEMPLATE() GPUd() calink HitLinkUpData  ( const MEM_TYPE(AliHLTTPCCARow) &row, const calink &hitIndex ) const;
    MEM_TEMPLATE() GPUd() calink HitLinkDownData( const MEM_TYPE(AliHLTTPCCARow) &row, const calink &hitIndex ) const;

    MEM_TEMPLATE() GPUhd() GPUglobalref() const cahit2 *HitData( const MEM_TYPE(AliHLTTPCCARow) &row ) const {return &fHitData[row.fHitNumberOffset];}
    GPUhd() GPUglobalref() const cahit2* HitData() const { return(fHitData); }
	MEM_TEMPLATE() GPUd() GPUglobalref() const calink *HitLinkUpData  ( const MEM_TYPE(AliHLTTPCCARow) &row ) const {return &fLinkUpData[row.fHitNumberOffset];}
	MEM_TEMPLATE() GPUd() GPUglobalref() const calink *HitLinkDownData( const MEM_TYPE(AliHLTTPCCARow) &row ) const {return &fLinkDownData[row.fHitNumberOffset];}
	MEM_TEMPLATE() GPUd() GPUglobalref() const calink *FirstHitInBin( const MEM_TYPE( AliHLTTPCCARow) &row ) const {return &fFirstHitInBin[row.fFirstHitInBinOffset];}

    MEM_TEMPLATE() GPUd() void SetHitLinkUpData  ( const MEM_TYPE(AliHLTTPCCARow) &row, const calink &hitIndex,
                             const calink &value );
    MEM_TEMPLATE() GPUd() void SetHitLinkDownData( const MEM_TYPE(AliHLTTPCCARow) &row, const calink &hitIndex,
                             const calink &value );

    /**
     * Reset all links to -1.
     */
    void ClearLinks();

    /**
     * Return the y and z coordinate(s) of the given hit(s).
     */
    // TODO return float_v
    MEM_TEMPLATE() GPUd() cahit HitDataY( const MEM_TYPE( AliHLTTPCCARow) &row, const uint_v &hitIndex ) const;
    MEM_TEMPLATE() GPUd() cahit HitDataZ( const MEM_TYPE( AliHLTTPCCARow) &row, const uint_v &hitIndex ) const;
    MEM_TEMPLATE() GPUd() cahit2 HitData( const MEM_TYPE( AliHLTTPCCARow) &row, const uint_v &hitIndex ) const;

    /**
     * For a given bin index, content tells how many hits there are in the preceding bins. This maps
     * directly to the hit index in the given row.
     *
     * \param binIndexes in the range 0 to row.Grid.N + row.Grid.Ny + 3.
     */
    MEM_TEMPLATE() GPUd() calink FirstHitInBin( const MEM_TYPE( AliHLTTPCCARow)&row, calink binIndexes ) const;

    /**
     * If the given weight is higher than what is currently stored replace with the new weight.
     */
    MEM_TEMPLATE() GPUd() void MaximizeHitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex, int_v weight );
	MEM_TEMPLATE() GPUd() void SetHitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex, int_v weight );

    /**
     * Return the maximal weight the given hit got from one tracklet
     */
    MEM_TEMPLATE() GPUd() int_v HitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex ) const;

    /**
     * Reset all hit weights to 0.
     */
    void ClearHitWeights();

    /**
     * Returns the index in the original AliHLTTPCCAClusterData object of the given hit
     */
    MEM_TEMPLATE() GPUhd() int_v ClusterDataIndex( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex ) const;

    /**
     * Return the row object for the given row index.
     */
	GPUhd() GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow)& Row( int rowIndex ) const {return fRows[rowIndex];}
    GPUhd() GPUglobalref() MEM_GLOBAL(AliHLTTPCCARow)* Rows() const {return fRows;}

    GPUhd() GPUglobalref() int* HitWeights() const {return(fHitWeights); }

    GPUhd() void SetGPUTextureBase(char* const val) {fGPUTextureBase = val;}
    GPUhd() char* GPUTextureBase() const { return(fGPUTextureBase); }
    GPUhd() char* GPUTextureBaseConst() const { return(fGPUTextureBase); }

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
	GPUh() char* Memory() const {return(fMemory); }
    GPUh() size_t MemorySize() const {return(fMemorySize); }
    GPUh() size_t GpuMemorySize() const {return(fGpuMemorySize); }
    GPUh() int GPUSharedDataReq() const { return fGPUSharedDataReq; }
#endif

    void SetGpuSliceData() { fIsGpuSliceData = 1; }
    float MaxZ() const { return fMaxZ; }

  private:
    AliHLTTPCCASliceData( const AliHLTTPCCASliceData & );
    AliHLTTPCCASliceData& operator=( const AliHLTTPCCASliceData & ) ;

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
    void CreateGrid( AliHLTTPCCARow *row, const float2* data, int ClusterDataHitNumberOffset );
    int PackHitData( AliHLTTPCCARow *row, const AliHLTArray<AliHLTTPCCAHit, 1> &binSortedHits );
#endif

    int fIsGpuSliceData;       //Slice Data for GPU Tracker?
    int fGPUSharedDataReq;     //Size of shared memory required for GPU Reconstruction

    int fFirstRow;             //First non-empty row
    int fLastRow;              //Last non-empty row

    int fNumberOfHits;         // the number of hits in this slice
    int fNumberOfHitsPlusAlign;
    
    float fMaxZ;

    size_t fMemorySize;           // size of the allocated memory in bytes
    size_t fGpuMemorySize;        // size of Memory needed to be transfered to GPU
    GPUglobalref() char *fMemory;             // pointer to the allocated memory where all the following arrays reside in
    GPUglobalref() char *fGPUTextureBase;     // pointer to start of GPU texture

    GPUglobalref() MEM_GLOBAL(AliHLTTPCCARow) *fRows;     // The row objects needed for most accessor functions

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

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline calink MEM_LG(AliHLTTPCCASliceData)::HitLinkUpData  ( const MEM_TYPE( AliHLTTPCCARow)&row, const calink &hitIndex ) const
{
  return fLinkUpData[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline calink MEM_LG(AliHLTTPCCASliceData)::HitLinkDownData( const MEM_TYPE( AliHLTTPCCARow)&row, const calink &hitIndex ) const
{
  return fLinkDownData[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline void MEM_LG(AliHLTTPCCASliceData)::SetHitLinkUpData  ( const MEM_TYPE( AliHLTTPCCARow)&row, const calink &hitIndex, const calink &value )
{
  fLinkUpData[row.fHitNumberOffset + hitIndex] = value;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline void MEM_LG(AliHLTTPCCASliceData)::SetHitLinkDownData( const MEM_TYPE( AliHLTTPCCARow)&row, const calink &hitIndex, const calink &value )
{
  fLinkDownData[row.fHitNumberOffset + hitIndex] = value;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline cahit MEM_LG(AliHLTTPCCASliceData)::HitDataY( const MEM_TYPE( AliHLTTPCCARow)&row, const uint_v &hitIndex ) const
{
  return fHitData[row.fHitNumberOffset + hitIndex].x;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline cahit MEM_LG(AliHLTTPCCASliceData)::HitDataZ( const MEM_TYPE( AliHLTTPCCARow)&row, const uint_v &hitIndex ) const
{
  return fHitData[row.fHitNumberOffset + hitIndex].y;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline cahit2 MEM_LG(AliHLTTPCCASliceData)::HitData( const MEM_TYPE( AliHLTTPCCARow)&row, const uint_v &hitIndex ) const
{
  return fHitData[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline calink MEM_LG(AliHLTTPCCASliceData)::FirstHitInBin( const MEM_TYPE( AliHLTTPCCARow)&row, calink binIndexes ) const
{
  return fFirstHitInBin[row.fFirstHitInBinOffset + binIndexes];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUhd() inline int_v MEM_LG(AliHLTTPCCASliceData)::ClusterDataIndex( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex ) const
{
  return fClusterDataIndex[row.fHitNumberOffset + hitIndex];
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline void MEM_LG(AliHLTTPCCASliceData)::MaximizeHitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex, int_v weight )
{
  CAMath::AtomicMax( &fHitWeights[row.fHitNumberOffset + hitIndex], weight );
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline void MEM_LG(AliHLTTPCCASliceData)::SetHitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex, int_v weight )
{
  fHitWeights[row.fHitNumberOffset + hitIndex] = weight;
}

MEM_CLASS_PRE() MEM_TEMPLATE() GPUd() inline int_v MEM_LG(AliHLTTPCCASliceData)::HitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, uint_v hitIndex ) const
{
  return fHitWeights[row.fHitNumberOffset + hitIndex];
}

//typedef AliHLTTPCCASliceData SliceData;

#endif // ALIHLTTPCCASLICEDATA_H
