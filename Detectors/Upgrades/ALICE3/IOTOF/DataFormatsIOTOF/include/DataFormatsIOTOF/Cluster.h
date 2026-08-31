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

/// \file Cluster.h
/// \brief Definition of the IOTOF cluster
#ifndef ALICEO2_IOTOF_CLUSTER_H
#define ALICEO2_IOTOF_CLUSTER_H

#include <cstdint>
#include <string>
#include <iosfwd>
#include <Rtypes.h>

#include "Framework/Logger.h"

namespace o2
{
namespace iotof
{

/// Compact encoding for ALICE3 IOTOF cluster parameters inside a single 64-bit word.
struct ClusterInfo {
  // Bit widths (Total: 52 bits out of 64)
  static constexpr int NBitsRow      = 9;
  static constexpr int NBitsCol      = 8;
  static constexpr int NBitsRowSpan  = 4;
  static constexpr int NBitsColSpan  = 4;
  static constexpr int NBitsPattern  = 16;
  static constexpr int NBitsTopology = 11;

  // Bit offsets (ordered logically from LSB to MSB)
  static constexpr int ShiftRow      = 0;
  static constexpr int ShiftCol      = ShiftRow      + NBitsRow;      // 9
  static constexpr int ShiftRowSpan  = ShiftCol      + NBitsCol;      // 17
  static constexpr int ShiftColSpan  = ShiftRowSpan  + NBitsRowSpan;  // 21
  static constexpr int ShiftPattern  = ShiftColSpan  + NBitsColSpan;  // 25
  static constexpr int ShiftTopology = ShiftPattern  + NBitsPattern;  // 41

  // Bit masks
  static constexpr uint64_t MaskRow      = (1ULL << NBitsRow) - 1;
  static constexpr uint64_t MaskCol      = (1ULL << NBitsCol) - 1;
  static constexpr uint64_t MaskRowSpan  = (1ULL << NBitsRowSpan) - 1;
  static constexpr uint64_t MaskColSpan  = (1ULL << NBitsColSpan) - 1;
  static constexpr uint64_t MaskPattern  = (1ULL << NBitsPattern) - 1;
  static constexpr uint64_t MaskTopology = (1ULL << NBitsTopology) - 1;

  uint64_t data{0};

  // Constructors
  constexpr ClusterInfo() = default;
  constexpr ClusterInfo(uint64_t d) : data(d) {}

  // Static packer
  static constexpr uint64_t pack(uint32_t row, uint32_t col, uint32_t rowSpan, 
                                 uint32_t colSpan, uint32_t pattern, uint32_t topology) {
    return ((static_cast<uint64_t>(row)      & MaskRow)      << ShiftRow)      |
           ((static_cast<uint64_t>(col)      & MaskCol)      << ShiftCol)      |
           ((static_cast<uint64_t>(rowSpan)  & MaskRowSpan)  << ShiftRowSpan)  |
           ((static_cast<uint64_t>(colSpan)  & MaskColSpan)  << ShiftColSpan)  |
           ((static_cast<uint64_t>(pattern)  & MaskPattern)  << ShiftPattern)  |
           ((static_cast<uint64_t>(topology) & MaskTopology) << ShiftTopology);
  }

  // Getters
  constexpr uint32_t getRow()      const { return (data >> ShiftRow)      & MaskRow; }
  constexpr uint32_t getCol()      const { return (data >> ShiftCol)      & MaskCol; }
  constexpr uint32_t getRowSpan()  const { return (data >> ShiftRowSpan)  & MaskRowSpan; }
  constexpr uint32_t getColSpan()  const { return (data >> ShiftColSpan)  & MaskColSpan; }
  constexpr uint32_t getPattern()  const { return (data >> ShiftPattern)  & MaskPattern; }
  constexpr uint32_t getTopology() const { return (data >> ShiftTopology) & MaskTopology; }

  // Setters
  constexpr void setRow(uint32_t r) {
    data = (data & ~(MaskRow << ShiftRow)) | ((static_cast<uint64_t>(r) & MaskRow) << ShiftRow);
  }
  constexpr void setCol(uint32_t c) {
    data = (data & ~(MaskCol << ShiftCol)) | ((static_cast<uint64_t>(c) & MaskCol) << ShiftCol);
  }
  constexpr void setRowSpan(uint32_t rs) {
    data = (data & ~(MaskRowSpan << ShiftRowSpan)) | ((static_cast<uint64_t>(rs) & MaskRowSpan) << ShiftRowSpan);
  }
  constexpr void setColSpan(uint32_t cs) {
    data = (data & ~(MaskColSpan << ShiftColSpan)) | ((static_cast<uint64_t>(cs) & MaskColSpan) << ShiftColSpan);
  }
  constexpr void setPattern(uint32_t p) {
    data = (data & ~(MaskPattern << ShiftPattern)) | ((static_cast<uint64_t>(p) & MaskPattern) << ShiftPattern);
  }
  constexpr void setTopology(uint32_t t) {
    data = (data & ~(MaskTopology << ShiftTopology)) | ((static_cast<uint64_t>(t) & MaskTopology) << ShiftTopology);
  }

  ClassDefNV(ClusterInfo, 1);
};

class Cluster
{
 public:
  static constexpr uint16_t InvalidPatternID = static_cast<uint16_t>(ClusterInfo::MaskPattern);

  Cluster() = default;
  Cluster(UShort_t row, UShort_t col, UShort_t rowSpan, UShort_t colSpan, UShort_t patt, UShort_t topo, UShort_t chipID = 0, time_t time = 0.0f)
    : mChipID(chipID), mTime(time)
  {
    mClusterInfo.data = ClusterInfo::pack(row, col, rowSpan, colSpan, patt, topo);
  }

  void set(UShort_t row, UShort_t col, UShort_t rowSpan, UShort_t colSpan, UShort_t patt, UShort_t topo, UShort_t chipID, time_t time)
  {
    mClusterInfo.data = ClusterInfo::pack(row, col, rowSpan, colSpan, patt, topo);
    mChipID = chipID;
    mTime = time;
  }

  // Unpack Getters
  uint32_t getRow()      const { return mClusterInfo.getRow(); }
  uint32_t getCol()      const { return mClusterInfo.getCol(); }
  uint32_t getRowSpan()  const { return mClusterInfo.getRowSpan(); }
  uint32_t getColSpan()  const { return mClusterInfo.getColSpan(); }
  uint32_t getPattern()  const { return mClusterInfo.getPattern(); }
  uint32_t getTopology() const { return mClusterInfo.getTopology(); }
  int getSize() const {
    // Count the number of set bits in the pattern to determine the size of the cluster
    uint32_t pattern = getPattern();
    int size = 0;
    while (pattern) {
      size += pattern & 1;
      pattern >>= 1;
    }
    return size;
  }

  // BaseCluster / Interface Compatibility Getters
  uint32_t getChipID()   const { return mChipID; }
  uint32_t getSensorID() const { return mChipID; }
  time_t getTime()        const { return mTime; }
  uint64_t getPackedData() const { return mClusterInfo.data; }

  // Setters
  void setRow(UShort_t r)        { mClusterInfo.setRow(r); }
  void setCol(UShort_t c)        { mClusterInfo.setCol(c); }
  void setRowSpan(UShort_t rs)   { mClusterInfo.setRowSpan(rs); }
  void setColSpan(UShort_t cs)   { mClusterInfo.setColSpan(cs); }
  void setPatternID(UShort_t p)  { mClusterInfo.setPattern(p); }
  void setTopology(UShort_t t)   { mClusterInfo.setTopology(t); }
  void setChipID(UShort_t c)     { mChipID = c; }
  void setTime(time_t t)         { mTime = t; }

  // Operators & Debugging
  bool operator==(const Cluster& cl) const
  {
    return mClusterInfo.data == cl.mClusterInfo.data && mChipID == cl.mChipID && mTime == cl.mTime;
  }

  void print() const;
  std::string asString() const;

 private:
  ClusterInfo mClusterInfo{}; ///< 64-bit packed structure containing geometry/topology
  UShort_t mChipID{0};        ///< Chip / Sensor ID
  float mTime{0.0f};          ///< Hit timing information

  void sanityCheck();

  ClassDefNV(Cluster, 2);
};

} // namespace iotof
} // namespace o2

std::ostream& operator<<(std::ostream& stream, const o2::iotof::Cluster& cl);

#endif /* ALICEO2_IOTOF_CLUSTER_H */