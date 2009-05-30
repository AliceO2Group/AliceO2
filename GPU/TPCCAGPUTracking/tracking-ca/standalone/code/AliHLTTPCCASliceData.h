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
#include "AliHLTArray.h"

typedef int int_v;
typedef unsigned int uint_v;
typedef short short_v;
typedef unsigned short ushort_v;
typedef float float_v;

class AliHLTTPCCAClusterData;
template<typename T, int Dim> class AliHLTArray;
class AliHLTTPCCAHit;
class AliHLTTPCCAParam;

/**
 * Data abstraction class for the Slice Tracker.
 *
 * Different architectures implement this for the most efficient loads and stores. All access to the
 * data happens through inline functions so that access to the data has no extra costs.
 */
class AliHLTTPCCASliceData
{
	friend class AliHLTTPCCAGPUTracker;
  public:
    AliHLTTPCCASliceData()
        : fNumberOfHits( 0 ), fMemorySize( 0 ), fMemory( 0 ), fLinkUpData( 0 ),
        fLinkDownData( 0 ), fHitDataY( 0 ), fHitDataZ( 0 ), fClusterDataIndex( 0 ),
        fFirstHitInBin( 0 ), fHitWeights( 0 ) {}

    void InitializeRows( const AliHLTTPCCAParam &parameters );

    /**
     * (Re)Create the data that is tuned for optimal performance of the algorithm from the cluster
     * data.
     */

	char* SetGPUSliceDataMemory(char* pGPUMemory, const AliHLTTPCCAClusterData *data);
	size_t SetPointers(const AliHLTTPCCAClusterData *data, bool allocate = false);
    void InitFromClusterData( const AliHLTTPCCAClusterData &data );

    /**
     * Clear the slice data (e.g. for an empty slice)
     */
    void Clear();

    /**
     * Return the number of hits in this slice.
     */
    GPUhd() int NumberOfHits() const { return fNumberOfHits; }

    /**
     * Access to the hit links.
     *
     * The links values give the hit index in the row above/below. Or -1 if there is no link.
     */
    short_v HitLinkUpData  ( const AliHLTTPCCARow &row, const short_v &hitIndex ) const;
    short_v HitLinkDownData( const AliHLTTPCCARow &row, const short_v &hitIndex ) const;
    void SetHitLinkUpData  ( const AliHLTTPCCARow &row, const short_v &hitIndex,
                             const short_v &value );
    void SetHitLinkDownData( const AliHLTTPCCARow &row, const short_v &hitIndex,
                             const short_v &value );

    /**
     * Reset all links to -1.
     */
    void ClearLinks();

    /**
     * Return the y and z coordinate(s) of the given hit(s).
     */
    // TODO return float_v
    short_v HitDataY( const AliHLTTPCCARow &row, const uint_v &hitIndex ) const;
    short_v HitDataZ( const AliHLTTPCCARow &row, const uint_v &hitIndex ) const;

    /**
     * For a given bin index, content tells how many hits there are in the preceding bins. This maps
     * directly to the hit index in the given row.
     *
     * \param binIndexes in the range 0 to row.Grid.N + row.Grid.Ny + 3.
     */
    ushort_v FirstHitInBin( const AliHLTTPCCARow &row, ushort_v binIndexes ) const;

    /**
     * If the given weight is higher than what is currently stored replace with the new weight.
     */
    void MaximizeHitWeight( const AliHLTTPCCARow &row, uint_v hitIndex, int_v weight );

    /**
     * Return the maximal weight the given hit got from one tracklet
     */
    int_v HitWeight( const AliHLTTPCCARow &row, uint_v hitIndex ) const;

    /**
     * Reset all hit weights to 0.
     */
    void ClearHitWeights();

    /**
     * Returns the index in the original AliHLTTPCCAClusterData object of the given hit
     */
    int_v ClusterDataIndex( const AliHLTTPCCARow &row, uint_v hitIndex ) const;

    /**
     * Return the row object for the given row index.
     */
    const AliHLTTPCCARow &Row( int rowIndex ) const;

  private:

    AliHLTTPCCASliceData( const AliHLTTPCCASliceData & )
        : fNumberOfHits( 0 ), fMemorySize( 0 ), fMemory( 0 ), fLinkUpData( 0 ),
        fLinkDownData( 0 ), fHitDataY( 0 ), fHitDataZ( 0 ), fClusterDataIndex( 0 ),
        fFirstHitInBin( 0 ), fHitWeights( 0 ) {}

    AliHLTTPCCASliceData& operator=( const AliHLTTPCCASliceData & ) {
      return *this;
    }

    void CreateGrid( AliHLTTPCCARow *row, const AliHLTTPCCAClusterData &data );
    void PackHitData( AliHLTTPCCARow *row, const AliHLTArray<AliHLTTPCCAHit, 1> &binSortedHits );

    AliHLTTPCCARow fRows[200]; // The row objects needed for most accessor functions

    int fNumberOfHits;         // the number of hits in this slice
    int fMemorySize;           // size of the allocated memory in bytes
    char *fMemory;             // pointer to the allocated memory where all the following arrays reside in

    short *fLinkUpData;        // hit index in the row above which is linked to the given (global) hit index
    short *fLinkDownData;      // hit index in the row below which is linked to the given (global) hit index

    unsigned short *fHitDataY;         // packed y coordinate of the given (global) hit index
    unsigned short *fHitDataZ;         // packed z coordinate of the given (global) hit index

    int *fClusterDataIndex;    // see ClusterDataIndex()

    /*
     * The size of the array is row.Grid.N + row.Grid.Ny + 3. The row.Grid.Ny + 3 is an optimization
     * to remove the need for bounds checking. The last values are the same as the entry at [N - 1].
     */
    unsigned short *fFirstHitInBin;         // see FirstHitInBin

    int *fHitWeights;          // the weight of the longest tracklet crossed the cluster

};

GPUd() inline short_v AliHLTTPCCASliceData::HitLinkUpData  ( const AliHLTTPCCARow &row, const short_v &hitIndex ) const
{
  return fLinkUpData[row.fHitNumberOffset + hitIndex];
}

GPUd() inline short_v AliHLTTPCCASliceData::HitLinkDownData( const AliHLTTPCCARow &row, const short_v &hitIndex ) const
{
  return fLinkDownData[row.fHitNumberOffset + hitIndex];
}

GPUd() inline void AliHLTTPCCASliceData::SetHitLinkUpData  ( const AliHLTTPCCARow &row, const short_v &hitIndex, const short_v &value )
{
  fLinkUpData[row.fHitNumberOffset + hitIndex] = value;
}

GPUd() inline void AliHLTTPCCASliceData::SetHitLinkDownData( const AliHLTTPCCARow &row, const short_v &hitIndex, const short_v &value )
{
  fLinkDownData[row.fHitNumberOffset + hitIndex] = value;
}

GPUd() inline short_v AliHLTTPCCASliceData::HitDataY( const AliHLTTPCCARow &row, const uint_v &hitIndex ) const
{
  return fHitDataY[row.fHitNumberOffset + hitIndex];
}

GPUd() inline short_v AliHLTTPCCASliceData::HitDataZ( const AliHLTTPCCARow &row, const uint_v &hitIndex ) const
{
  return fHitDataZ[row.fHitNumberOffset + hitIndex];
}

GPUd() inline ushort_v AliHLTTPCCASliceData::FirstHitInBin( const AliHLTTPCCARow &row, ushort_v binIndexes ) const
{
  return fFirstHitInBin[row.fFirstHitInBinOffset + binIndexes];
}

GPUhd() inline int_v AliHLTTPCCASliceData::ClusterDataIndex( const AliHLTTPCCARow &row, uint_v hitIndex ) const
{
  return fClusterDataIndex[row.fHitNumberOffset + hitIndex];
}

GPUhd() inline const AliHLTTPCCARow &AliHLTTPCCASliceData::Row( int rowIndex ) const
{
  return fRows[rowIndex];
}

GPUd() inline void AliHLTTPCCASliceData::MaximizeHitWeight( const AliHLTTPCCARow &row, uint_v hitIndex, int_v weight )
{
  CAMath::AtomicMax( &fHitWeights[row.fHitNumberOffset + hitIndex], weight );
}

GPUd() inline int_v AliHLTTPCCASliceData::HitWeight( const AliHLTTPCCARow &row, uint_v hitIndex ) const
{
  return fHitWeights[row.fHitNumberOffset + hitIndex];
}

typedef AliHLTTPCCASliceData SliceData;

#endif // SLICEDATA_H
