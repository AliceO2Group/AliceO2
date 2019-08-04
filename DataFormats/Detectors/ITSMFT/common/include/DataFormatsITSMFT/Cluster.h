// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.h
/// \brief Definition of the ITSMFT cluster
#ifndef ALICEO2_ITSMFT_CLUSTER_H
#define ALICEO2_ITSMFT_CLUSTER_H

#include "ReconstructionDataFormats/BaseCluster.h"

// uncomment this to have cluster topology stored
#define _ClusterTopology_

#define CLUSTER_VERSION 3

namespace o2
{
namespace itsmft
{
class GeometryTGeo;
/// \class Cluster
/// \brief Cluster class for the ITSMFT
///

class Cluster : public o2::BaseCluster<float>
{
 public:
  enum { // frame in which the track is currently defined
    kUsed,
    kShared
  };
  //
  enum {
    kOffsNZ = 0,
    kMaskNZ = 0xff,
    kOffsNX = 8,
    kMaskNX = 0xff,
    kOffsNPix = 16,
    kMaskNPix = 0x1ff,
    kOffsClUse = 25,
    kMaskClUse = 0x7f
  };
  //
  // used for the cluster topology definition
  enum {
    kMaxPatternBits = 32 * 16,
    kMaxPatternBytes = kMaxPatternBits / 8,
    kSpanMask = 0x7fff,
    kTruncateMask = 0x8000
  };

  using BaseCluster::BaseCluster;

 public:
  static constexpr int maxLabels = 10;

  ~Cluster() = default;

  Cluster& operator=(const Cluster& cluster) = delete; // RS why?

  //****** Basic methods ******************
  void setUsed() { setBit(kUsed); }
  void setShared() { setBit(kShared); }
  void increaseClusterUsage() { isUsed() ? setBit(kShared) : setBit(kUsed); }
  //
  bool isUsed() const { return isBitSet(kUsed); }
  bool isShared() const { return isBitSet(kShared); }
  //
  void setNxNzN(UChar_t nx, UChar_t nz, UShort_t n)
  {
    mNxNzN = ((n & kMaskNPix) << kOffsNPix) + ((nx & kMaskNX) << kOffsNX) + ((nz & kMaskNZ) << kOffsNZ);
  }
  void setClusterUsage(int n);
  void modifyClusterUsage(bool used = kTRUE) { used ? incClusterUsage() : decreaseClusterUsage(); }
  void incClusterUsage()
  {
    setClusterUsage(getClusterUsage() + 1);
    increaseClusterUsage();
  }
  void decreaseClusterUsage();
  int getNx() const { return (mNxNzN >> kOffsNX) & kMaskNX; }
  int getNz() const { return (mNxNzN >> kOffsNZ) & kMaskNZ; }
  int getNPix() const { return (mNxNzN >> kOffsNPix) & kMaskNPix; }
  int getClusterUsage() const { return (mNxNzN >> kOffsClUse) & kMaskClUse; }
  //
  UInt_t getROFrame() const { return mROFrame; }
  void setROFrame(UInt_t v) { mROFrame = v; }

  // bool hasCommonTrack(const Cluster* cl) const;
  //
  void print() const;

#ifdef _ClusterTopology_
  bool testPixel(UShort_t row, UShort_t col) const;
  void resetPattern();
  int getPatternRowSpan() const { return mPatternNRows & kSpanMask; }
  int getPatternColSpan() const { return mPatternNCols & kSpanMask; }
  bool isPatternRowsTruncated() const { return mPatternNRows & kTruncateMask; }
  bool isPatternColsTruncated() const { return mPatternNCols & kTruncateMask; }
  bool isPatternTruncated() const { return isPatternRowsTruncated() || isPatternColsTruncated(); }
  void setPatternRowMin(UShort_t row) { mPatternRowMin = row; }
  void setPatternColMin(UShort_t col) { mPatternColMin = col; }
  void getPattern(void* destination, int nbytes) const { memcpy(destination, mPattern, nbytes); }
  int getPatternRowMin() const { return mPatternRowMin; }
  int getPatternColMin() const { return mPatternColMin; }

  ///< set pattern span in rows, flag if truncated
  void setPatternRowSpan(UShort_t nr, bool truncated)
  {
    mPatternNRows = kSpanMask & nr;
    if (truncated) {
      mPatternNRows |= kTruncateMask;
    }
  }

  ///< set pattern span in columns, flag if truncated
  void setPatternColSpan(UShort_t nc, bool truncated)
  {
    mPatternNCols = kSpanMask & nc;
    if (truncated) {
      mPatternNCols |= kTruncateMask;
    }
  }

  ///< fire the pixel in the pattern, no check for the overflow, must be done in advance!
  void setPixel(UShort_t row, UShort_t col)
  {
    int nbits = row * getPatternColSpan() + col;
    mPattern[nbits >> 3] |= (0x1 << (7 - (nbits % 8)));
  }

  ///< fire the pixel in the pattern, no check for the overflow, must be done in advance!
  void unSetPixel(UShort_t row, UShort_t col)
  {
    int nbits = row * getPatternColSpan() + col;
    mPattern[nbits >> 3] &= (0xff ^ (0x1 << (nbits % 8)));
  }

#endif
  //
 protected:
  //
  UInt_t mROFrame; ///< RO Frame

  Int_t mNxNzN = 0; ///< effective cluster size in X (1st byte) and Z (2nd byte) directions
                    ///< and total Npix(next 9 bits).
                    ///> The last 7 bits are used for clusters usage counter

#ifdef _ClusterTopology_
  UShort_t mPatternNRows = 0;               ///< pattern span in rows
  UShort_t mPatternNCols = 0;               ///< pattern span in columns
  UShort_t mPatternRowMin = 0;              ///< pattern start row
  UShort_t mPatternColMin = 0;              ///< pattern start column
  UChar_t mPattern[kMaxPatternBytes] = {0}; ///< cluster topology
  //
  ClassDefNV(Cluster, CLUSTER_VERSION + 1);
#else
  ClassDefNV(Cluster, CLUSTER_VERSION);
#endif
};
//______________________________________________________
inline void Cluster::decreaseClusterUsage()
{
  // decrease cluster usage counter
  int n = getClusterUsage();
  if (n)
    setClusterUsage(--n);
  //
}

//______________________________________________________
inline void Cluster::setClusterUsage(Int_t n)
{
  // set cluster usage counter
  mNxNzN &= ~(kMaskClUse << kOffsClUse);
  mNxNzN |= (n & kMaskClUse) << kOffsClUse;
  if (n < 2)
    resetBit(kShared);
  if (!n)
    resetBit(kUsed);
}
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_CLUSTER_H */
