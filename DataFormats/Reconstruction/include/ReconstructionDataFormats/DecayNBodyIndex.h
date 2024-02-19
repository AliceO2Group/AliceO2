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

#ifndef ALICEO2_NBODY_INDEX_H
#define ALICEO2_NBODY_INDEX_H

#include "ReconstructionDataFormats/VtxTrackIndex.h"

namespace o2::dataformats
{

template <int N>
class DecayNBodyIndex
{
 public:
  using GIndex = o2::dataformats::VtxTrackIndex;
  DecayNBodyIndex() = default;
  DecayNBodyIndex(int v, const std::array<GIndex, N>& arr) : mVertexID(v), mProngIDs(arr) {}
  DecayNBodyIndex(int v, std::initializer_list<GIndex> l) : mVertexID(v)
  {
    assert(l.size() == N);
    int i = 0;
    for (auto e : l) {
      mProngIDs[i++] = e;
    }
  }
  GIndex getProngID(int i) const { return mProngIDs[i]; }
  void setProngID(int i, GIndex gid) { mProngIDs[i] = gid; }
  int getVertexID() const { return mVertexID; }
  void setVertexID(int id) { mVertexID = id; }
  uint8_t getBits() const { return mBits; }
  bool testBit(int i) const { return (mBits & (0x1 << i)) != 0; }
  void setBit(int i) { mBits |= (0x1 << i); }
  void resetBit(int i) { mBits &= ~(0x1 << i); }

  const std::array<GIndex, N>& getProngs() const { return mProngIDs; }
  static constexpr int getNProngs() { return N; }

 protected:
  int mVertexID = -1;                // id of parent vertex
  std::array<GIndex, N> mProngIDs{}; // global IDs of prongs
  uint8_t mBits = 0;                 // user defined bits

  ClassDefNV(DecayNBodyIndex, 2);
};

class V0Index : public DecayNBodyIndex<2>
{
 public:
  using DecayNBodyIndex<2>::DecayNBodyIndex;
  V0Index(int v, GIndex p, GIndex n) : DecayNBodyIndex<2>(v, {p, n}) {}
  bool isStandaloneV0() const { return testBit(0); }
  bool isPhotonOnly() const { return testBit(1); }
  void setStandaloneV0() { setBit(0); }
  void setPhotonOnly() { setBit(1); }
  ClassDefNV(V0Index, 1);
};

class Decay3BodyIndex : public DecayNBodyIndex<3>
{
 public:
  using DecayNBodyIndex<3>::DecayNBodyIndex;
  Decay3BodyIndex(int v, GIndex p0, GIndex p1, GIndex p2) : DecayNBodyIndex<3>(v, {p0, p1, p2}) {}
  ClassDefNV(Decay3BodyIndex, 1);
};

class CascadeIndex
{
 public:
  using GIndex = o2::dataformats::VtxTrackIndex;
  CascadeIndex() = default;
  CascadeIndex(int v, int v0id, GIndex bachelorID) : mVertexID(v), mV0ID(v0id), mBach(bachelorID) {}

  GIndex getBachelorID() const { return mBach; }
  void setBachelorID(GIndex gid) { mBach = gid; }

  int getV0ID() const { return mV0ID; }
  void setV0ID(int vid) { mV0ID = vid; }

  int getVertexID() const { return mVertexID; }
  void setVertexID(int id) { mVertexID = id; }

 protected:
  int mVertexID = -1;
  int mV0ID = -1;
  GIndex mBach{};

  ClassDefNV(CascadeIndex, 1);
};

} // namespace o2::dataformats

#endif
