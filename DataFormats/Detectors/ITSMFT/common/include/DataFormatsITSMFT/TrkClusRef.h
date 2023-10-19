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

/// \file TrkClusRef.h
/// \brief Reference on ITS/MFT clusters set

#ifndef ALICEO2_ITSMFT_TRKCLUSREF_H
#define ALICEO2_ITSMFT_TRKCLUSREF_H

#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace itsmft
{

// can refer to max 15 indices in the vector of total length <268435456, i.e. 17895697 tracks in worst case
struct TrkClusRef : public o2::dataformats::RangeRefComp<4> {
  using o2::dataformats::RangeRefComp<4>::RangeRefComp;
  uint32_t clsizes = 0; ///< cluster sizes for each layer
  uint16_t pattern = 0; ///< layers pattern

  GPUd() int getNClusters() const { return getEntries(); }
  bool hasHitOnLayer(int i) { return pattern & (0x1 << i); }

  void setClusterSize(int l, int size)
  {
    if (l >= 8)
      return;
    if (size > 15)
      size = 15;
    clsizes &= ~(0xf << (l * 4));
    clsizes |= (size << (l * 4));
  }

  int getClusterSize(int l)
  {
    if (l >= 8)
      return 0;
    return (clsizes >> (l * 4)) & 0xf;
  }

  int getClusterSizes() const
  {
    return clsizes;
  }

  ClassDefNV(TrkClusRef, 2);
};

} // namespace itsmft
} // namespace o2

#endif
