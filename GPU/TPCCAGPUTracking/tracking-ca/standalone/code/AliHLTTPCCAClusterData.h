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

#ifndef ALIHLTTPCCACLUSTERDATA_H
#define ALIHLTTPCCACLUSTERDATA_H

#include "AliHLTTPCCADef.h"
#include <iostream>
#include <vector>

/**
 * Cluster data which keeps history about changes
 *
 * The algorithm doesn't work on this data. Instead the AliHLTTPCCASliceData is created from this.
 */
class AliHLTTPCCAClusterData
{
  public:

    AliHLTTPCCAClusterData(): fSliceIndex( 0 ), fData( NULL ), fNumberOfClusters(0), fAllocated(0) {}
    ~AliHLTTPCCAClusterData();

    struct Data {
      int fId;
      int fRow;
      float fX;
      float fY;
      float fZ;
      float fAmp;
    };

    /**
     * prepare for the reading of event
     */
    void StartReading( int sliceIndex, int guessForNumberOfClusters = 256 );

    /**
     *  read next cluster
     */
    void ReadCluster( int id, int iRow, float x, float y, float z, float amp ) {
	  if (fNumberOfClusters >= fAllocated) Allocate(fNumberOfClusters + 64);
      Data d = { id, iRow, x, y, z, amp};
      fData[fNumberOfClusters++] = d;
    }

	Data* Clusters() { return(fData); }
	void SetNumberOfClusters(int number) {fNumberOfClusters = number;}

    /**
     * Read/Write Events from/to file
     */
    void ReadEvent(std::istream &in);
    void WriteEvent(std::ostream &out) const;
	template <class T> void ReadEventVector(T* &data, std::istream &in, int MinSize = 0);
    template <class T> void WriteEventVector(const T* const &data, std::ostream &out) const;

    /**
     * "remove" one cluster and "add" two new ones, keeping history.
     */
    //void Split( int index, /* TODO: need some parameters how to split */ );

    // TODO: some access to history of merges and splits

    /**
     * The slice index this data belongs to
     */
    int SliceIndex() const { return fSliceIndex; }

    /**
     * Return the number of clusters in this slice.
     */
    int NumberOfClusters() const { return (int) fNumberOfClusters; }

    /**
     * Return the x coordinate of the given cluster.
     */
    float X( int index ) const { return fData[index].fX; }

    /**
     * Return the y coordinate of the given cluster.
     */
    float Y( int index ) const { return fData[index].fY; }

    /**
     * Return the z coordinate of the given cluster.
     */
    float Z( int index ) const { return fData[index].fZ; }

    /**
     * Return the amplitude of the given cluster.
     */
    float Amp( int index ) const { return fData[index].fAmp; }

    /**
     * Return the global ID of the given cluster.
     */
    int Id( int index ) const { return fData[index].fId; }

    /**
     * Return the row number/index of the given cluster.
     */
    int RowNumber( int index ) const { return fData[index].fRow; }

    Data *GetClusterData( int index ) { return &( fData[index] ); }

  private:
	AliHLTTPCCAClusterData(AliHLTTPCCAClusterData&): fSliceIndex( 0 ), fData( NULL ), fNumberOfClusters(0), fAllocated(0) {}
	AliHLTTPCCAClusterData& operator=( const AliHLTTPCCAClusterData& );

    /** TODO
     * "remove" two clusters and "add" a new one, keeping history.
     */
    void Merge( int index1, int index2 );

	void Allocate( int number);
	static bool CompareClusters( const Data &a, const Data &b ) { return ( a.fRow == b.fRow ? (a.fY < b.fY) : (a.fRow < b.fRow) ); }

    int fSliceIndex;  // the slice index this data belongs to
    Data* fData; // list of data of clusters
	int fNumberOfClusters;	//Current number of clusters stored in fData
	int fAllocated; //Number of clusters that can be stored in fData
};

typedef AliHLTTPCCAClusterData ClusterData;

#endif // CLUSTERDATA_H
