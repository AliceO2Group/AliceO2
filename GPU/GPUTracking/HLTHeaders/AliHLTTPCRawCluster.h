// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliHLTTPCRawCluster.h
/// \author ALICE HLT Project

#ifndef ALIHLTTPCRAWCLUSTER_H
#define ALIHLTTPCRAWCLUSTER_H

/**
 * @struct AliHLTTPCRawCluster
 * Primitive data of a TPC cluster in raw coordinates. The plan is to store the
 * data in a compressed format by limiting the resolution of the float values.
 * @ingroup alihlt_tpc_datastructs
 */
struct AliHLTTPCRawCluster {
  AliHLTTPCRawCluster() : fPadRow(0), mFlags(0), fPad(0.), fTime(0.), fSigmaPad2(0.), fSigmaTime2(0.), fCharge(0), fQMax(0) {}

  AliHLTTPCRawCluster(int16_t PadRow, float Pad, float Time, float SigmaPad2, float SigmaTime2, uint16_t Charge, uint16_t QMax, uint16_t Flags) : fPadRow(PadRow), mFlags(Flags), fPad(Pad), fTime(Time), fSigmaPad2(SigmaPad2), fSigmaTime2(SigmaTime2), fCharge(Charge), fQMax(QMax) {}

  AliHLTTPCRawCluster(const AliHLTTPCRawCluster& other) : fPadRow(other.fPadRow), mFlags(other.mFlags), fPad(other.fPad), fTime(other.fTime), fSigmaPad2(other.fSigmaPad2), fSigmaTime2(other.fSigmaTime2), fCharge(other.fCharge), fQMax(other.fQMax) {} // NOLINT

  AliHLTTPCRawCluster& operator=(const AliHLTTPCRawCluster& other)
  {
    if (this == &other) {
      return *this;
    }
    this->~AliHLTTPCRawCluster();
    new (this) AliHLTTPCRawCluster(other);
    return *this;
  }

  void Clear()
  {
    this->~AliHLTTPCRawCluster();
    new (this) AliHLTTPCRawCluster;
  }

  int16_t fPadRow;
  uint16_t mFlags; // Flags: (1 << 0): Split in pad direction
                   //       (1 << 1): Split in time direction
                   //       (1 << 2): Edge Cluster
                   // During cluster merging, flags are OR'd
  float fPad;
  float fTime;
  float fSigmaPad2;
  float fSigmaTime2;
  uint16_t fCharge;
  uint16_t fQMax;

  int32_t GetPadRow() const { return fPadRow; }
  float GetPad() const { return fPad; }
  float GetTime() const { return fTime; }
  float GetSigmaPad2() const { return fSigmaPad2; }
  float GetSigmaTime2() const { return fSigmaTime2; }
  int32_t GetCharge() const { return fCharge; }
  int32_t GetQMax() const { return fQMax; }
  bool GetFlagSplitPad() const { return (mFlags & (1 << 0)); }
  bool GetFlagSplitTime() const { return (mFlags & (1 << 1)); }
  bool GetFlagSplitAny() const { return (mFlags & 3); }
  bool GetFlagEdge() const { return (mFlags & (1 << 2)); }
  bool GetFlagSplitAnyOrEdge() const { return (mFlags & 7); }
  uint16_t GetFlags() const { return (mFlags); }

  void SetPadRow(int16_t padrow) { fPadRow = padrow; }
  void SetPad(float pad) { fPad = pad; }
  void SetTime(float time) { fTime = time; }
  void SetSigmaPad2(float sigmaPad2) { fSigmaPad2 = sigmaPad2; }
  void SetSigmaTime2(float sigmaTime2) { fSigmaTime2 = sigmaTime2; }
  void SetCharge(uint16_t charge) { fCharge = charge; }
  void SetQMax(uint16_t qmax) { fQMax = qmax; }

  void ClearFlags() { mFlags = 0; }
  void SetFlags(uint16_t flags) { mFlags = flags; }
  void SetFlagSplitPad() { mFlags |= (1 << 0); }
  void SetFlagSplitTime() { mFlags |= (1 << 1); }
  void SetFlagEdge() { mFlags |= (1 << 2); }
};
typedef struct AliHLTTPCRawCluster AliHLTTPCRawCluster;

struct AliHLTTPCRawClusterData {
  uint32_t fVersion; // version number
  uint32_t fCount;   // number of clusters
#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTPCRawCluster fClusters[1]; // array of clusters
#else
  AliHLTTPCRawCluster fClusters[0]; // array of clusters
#endif
};
typedef struct AliHLTTPCRawClusterData AliHLTTPCRawClusterData;

#endif
