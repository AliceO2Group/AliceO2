// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PackedCharge.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_ARRAY2D_H
#define O2_GPU_ARRAY2D_H

#include "clusterFinderDefs.h"
#include "ChargePos.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <typename T, typename Layout>
class AbstractArray2D
{

 public:
  GPUdi() explicit AbstractArray2D(T* d) : data(d) {}

  GPUdi() T& operator[](const ChargePos& p) { return data[Layout::idx(p)]; }
  GPUdi() const T& operator[](const ChargePos& p) const { return data[Layout::idx(p)]; }

  GPUdi() void safeWrite(const ChargePos& p, const T& v)
  {
    if (data != nullptr) {
      (*this)[p] = v;
    }
  }

 private:
  T* data;
};

template <typename Grid>
class TilingLayout
{
 public:
  enum {
    Height = Grid::Height,
    Width = Grid::Width,
  };

  GPUdi() static size_t idx(const ChargePos& p)
  {
    const size_t widthInTiles = (TPC_NUM_OF_PADS + Width - 1) / Width;

    const size_t tilePad = p.gpad / Width;
    const size_t tileTime = p.time / Height;

    const size_t inTilePad = p.gpad % Width;
    const size_t inTileTime = p.time % Height;

    return (tileTime * widthInTiles + tilePad) * (Width * Height) + inTileTime * Width + inTilePad;
  }
};

template <typename T>
class LinearLayout
{
  GPUdi() static size_t idx(const ChargePos& p)
  {
    return TPC_NUM_OF_PADS * p.time + p.gpad;
  }
};

template <size_t S>
struct GridSize;

template <>
struct GridSize<1> {
  enum {
    Width = 8,
    Height = 8,
  };
};

template <>
struct GridSize<2> {
  enum {
    Width = 4,
    Height = 4,
  };
};

template <>
struct GridSize<4> {
  enum {
    Width = 4,
    Height = 4,
  };
};

#if defined(CHARGEMAP_TILING_LAYOUT)
template <typename T>
using Array2D = AbstractArray2D<T, TilingLayout<GridSize<sizeof(T)>>>;
#else
template <typename T>
using Array2D = AbstractArray2D<T, LinearLayout>;
#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
