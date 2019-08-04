// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-

#ifndef TPCRAWCLUSTER_H
#define TPCRAWCLUSTER_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   TPCRawCluster.h
//  @author Matthias Richter
//  @since  2015-09-27
//  @brief  ALICE HLT TPC raw cluster structure and tools

#include <iostream>
#include <fstream> // ifstream
#include <cstring> // memcpy

namespace o2
{
namespace AliceHLT
{

/**
 * @struct RawCluster
 * This is a redefinition from AliRoot/HLT/TPCLib/AliHLTTPCRawCluster.h for the
 * sake of reading HLT TPC raw cluster files into O2.
 *
 * TODO: there is no dependence on AliRoot, however, a test needs to be added
 * to check consistency if AliRoot is available in the build.
 */
struct RawCluster {

  int16_t GetPadRow() const { return fPadRow; }
  float GetPad() const { return fPad; }
  float GetTime() const { return fTime; }
  float GetSigmaPad2() const { return fSigmaPad2; }
  float GetSigmaTime2() const { return fSigmaTime2; }
  int32_t GetCharge() const { return fCharge; }
  int32_t GetQMax() const { return fQMax; }
  bool GetFlagSplitPad() const { return (fFlags & (1 << 0)); }
  bool GetFlagSplitTime() const { return (fFlags & (1 << 1)); }
  bool GetFlagSplitAny() const { return (fFlags & 3); }
  uint16_t GetFlags() const { return (fFlags); }

  int16_t fPadRow;
  uint16_t fFlags; //Flags: (1 << 0): Split in pad direction
                   //       (1 << 1): Split in time direction
                   //During cluster merging, flags are or'd
  float fPad;
  float fTime;
  float fSigmaPad2;
  float fSigmaTime2;
  uint16_t fCharge;
  uint16_t fQMax;
};

/**
 * @struct RawClusterData
 * Header data struct for a raw cluster block
 */
struct RawClusterData {
  uint32_t fVersion;       // version number
  uint32_t fCount;         // number of clusters
  RawCluster fClusters[0]; // array of clusters
};

std::ostream& operator<<(std::ostream& stream, const RawCluster& cluster)
{
  stream << "TPCRawCluster:"
         << " " << cluster.GetPadRow()
         << " " << cluster.GetPad()
         << " " << cluster.GetTime()
         << " " << cluster.GetSigmaPad2()
         << " " << cluster.GetSigmaTime2()
         << " " << cluster.GetCharge()
         << " " << cluster.GetQMax();
  return stream;
}

/**
 * @class RawClusterArray Wrapper to binary data block of HLT TPC raw clusters
 * Container class which provides access to the content of a binary block of
 * HLT TPC raw clusters.
 */
class RawClusterArray
{
 public:
  RawClusterArray() : mBuffer(nullptr), mBufferSize(0), mNClusters(0), mClusters(NULL), mClustersEnd(NULL) {}
  RawClusterArray(const char* filename) : mBuffer(nullptr), mBufferSize(0), mNClusters(0), mClusters(NULL), mClustersEnd(NULL)
  {
    init(filename);
  }
  RawClusterArray(unsigned char* buffer, int size) : mBuffer(nullptr), mBufferSize(0), mNClusters(0), mClusters(NULL), mClustersEnd(NULL)
  {
    init(buffer, size);
  }
  ~RawClusterArray() {}

  typedef uint8_t Buffer_t;

  int init(const char* filename)
  {
    std::ifstream input(filename, std::ifstream::binary);
    clear(0);
    if (input) {
      // get length of file:
      input.seekg(0, input.end);
      int length = input.tellg();
      input.seekg(0, input.beg);

      // allocate memory:
      mBuffer = new Buffer_t[length];
      mBufferSize = length;

      // read data as a block:
      input.read(reinterpret_cast<char*>(mBuffer), length);
      if (!input.good()) {
        clear(-1);
        std::cerr << "failed to read " << length << " byte(s) from file " << filename << std::endl;
      }

      input.close();
      return init();
    }
    std::cerr << "failed to open file " << filename << std::endl;
    return -1;
  }

  int init(unsigned char* buffer, int size)
  {
    if (!buffer || size <= 0)
      return -1;
    clear(0);
    mBuffer = new Buffer_t[size];
    mBufferSize = size;
    memcpy(mBuffer, buffer, size);
    return init();
  }

  int GetNClusters() const { return mNClusters; }

  RawCluster* begin() { return mClusters; }

  RawCluster* end() { return mClustersEnd; }

  RawCluster& operator[](int i)
  {
    if (i + 1 > mNClusters) {
      // runtime exeption?
      static RawCluster dummy;
      return dummy;
    }
    return *(mClusters + i);
  }

  void print() { print(std::cout); }

  template <typename StreamT>
  StreamT& print(StreamT& stream)
  {
    std::cout << "RawClusterArray: " << mNClusters << " cluster(s)" << std::endl;
    for (RawCluster* cluster = mClusters; cluster != mClustersEnd; cluster++) {
      std::cout << "  " << *cluster << std::endl;
    }
    return stream;
  }

 private:
  int init()
  {
    if (mBuffer == nullptr || mBufferSize == 0)
      return 0;
    if (mBufferSize < sizeof(RawClusterData))
      return -1;
    RawClusterData& clusterData = *reinterpret_cast<RawClusterData*>(mBuffer);

    if (clusterData.fCount * sizeof(RawCluster) + sizeof(RawClusterData) > mBufferSize) {
      std::cerr << "Format error, " << clusterData.fCount << " cluster(s) "
                << "would require "
                << (clusterData.fCount * sizeof(RawCluster) + sizeof(RawClusterData))
                << " byte(s), but only " << mBufferSize << " available" << std::endl;
      return clear(-1);
    }

    mNClusters = clusterData.fCount;
    mClusters = clusterData.fClusters;
    mClustersEnd = mClusters + mNClusters;

    return mNClusters;
  }

  int clear(int returnValue)
  {
    mNClusters = 0;
    mClusters = NULL;
    mClustersEnd = NULL;
    delete[] mBuffer;
    mBuffer = nullptr;
    mBufferSize = 0;

    return returnValue;
  }

  Buffer_t* mBuffer;
  int mBufferSize;
  int mNClusters;
  RawCluster* mClusters;
  RawCluster* mClustersEnd;
};

}; // namespace AliceHLT
}; // namespace o2
#endif
