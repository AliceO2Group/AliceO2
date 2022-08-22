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

/// @file  FlatHisto1D.h
/// \brief 1D messeageable histo class
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FLATHISTO1D_H
#define ALICEO2_FLATHISTO1D_H

#include <Rtypes.h>
#include <vector>
#include <gsl/span>
#include <type_traits>
#include <cassert>
#include <memory>

class TH1F;

namespace o2
{
namespace dataformats
{

/*
  Fast 1D histo class which can be messages as
  FlatHisto1D<float> histo(nbins, xmin, xmax);
  histo.fill(...);
  pc.outputs().snapshot(Output{"Origin", "Desc", 0, Lifetime::Timeframe}, histo.getBase());

  and received (read only!) as
  const auto hdata = pc.inputs().get<gsl::span<float>>("histodata");
  FlatHisto1D<float> histoView;
  histoView.adoptExternal(hdata);
  or directly
  FlatHisto1D<float> histoView(pc.inputs().get<gsl::span<float>>("histodata"));
*/

template <typename T = float>
class FlatHisto1D
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "T must be float or double");

 public:
  enum { NBins,
         XMin,
         XMax,
         BinSize,
         NServiceSlots };

  FlatHisto1D() = default;
  FlatHisto1D(uint32_t nb, T xmin, T xmax);
  FlatHisto1D(const FlatHisto1D& src);
  FlatHisto1D(const gsl::span<const T> ext) { adoptExternal(ext); }
  FlatHisto1D& operator=(const FlatHisto1D& rhs);
  void adoptExternal(const gsl::span<const T> ext);
  void init()
  {
    // when reading from file, need to call this method to make it operational
    assert(mContainer.size() > NServiceSlots);
    init(gsl::span<const T>(mContainer.data(), mContainer.size()));
  }
  void init(uint32_t nbx, T xmin, T xmax);
  uint32_t getNBins() const { return mNBins; }
  T getXMin() const { return mXMin; }
  T getXMax() const { return mXMax; }
  T getBinSize() const { return mBinSize; }
  T getBinSizeInv() const { return mBinSizeInv; }

  T getBinContent(uint32_t ib) const
  {
    assert(ib < getNBins());
    return mDataPtr[ib];
  }

  const T* getData() const
  {
    return mDataPtr;
  }

  T getBinContentForX(T x) const
  {
    auto bin = getBin(x);
    return isValidBin(bin) ? getBinContent(bin) : 0;
  }

  bool isValidBin(uint32_t bin) const { return bin < getNBins(); }
  bool isBinEmpty(uint32_t bin) const { return getBinContent(bin) == 0; }

  T getBinStart(uint32_t i) const
  {
    assert(i < getNBins());
    return getXMin() + i * getBinSize();
  }

  T getBinCenter(uint32_t i) const
  {
    assert(i < getNBins());
    return getXMin() + (i + 0.5) * getBinSize();
  }

  T getBinEnd(uint32_t i) const
  {
    assert(i < getNBins());
    return getXMin() + (i + 1) * getBinSize();
  }

  void add(const FlatHisto1D& other);

  void subtract(const FlatHisto1D& other);

  void setBinContent(uint32_t bin, T w)
  {
    assert(canFill() && isValidBin(bin));
    mDataPtr[bin] = w;
  }

  void clear()
  {
    assert(canFill());
    memset(mDataPtr, 0, sizeof(T) * getNBins());
  }

  T getSum() const;

  int fill(T x)
  {
    uint32_t bin = getBin(x);
    if (isValidBin(bin)) {
      mDataPtr[bin]++;
      return (int)bin;
    }
    return -1;
  }

  int fill(T x, T w)
  {
    uint32_t bin = getBin(x);
    if (isValidBin(bin)) {
      mDataPtr[bin] += w;
      return (int)bin;
    }
    return -1;
  }

  void fillBin(uint32_t bin, T w)
  {
    if (isValidBin(bin)) {
      mDataPtr[bin] += w;
    }
  }

  uint32_t getBin(T x) const
  {
    auto dx = x - getXMin();
    return dx < 0 ? 0xffffffff : uint32_t(dx * getBinSizeInv());
  }

  bool canFill() const
  {
    // histo can be filled only if hase its own data, otherwise only query can be done on the view
    return mContainer.size() > NServiceSlots;
  }

  std::unique_ptr<TH1F> createTH1F(const std::string& name = "histo1d") const;

  const std::vector<T>& getBase() const { return mContainer; }
  gsl::span<const T> getView() const { return mContainerView; }

 protected:
  void init(const gsl::span<const T> ext);

  std::vector<T> mContainer;           //    global container
  gsl::span<const T> mContainerView{}; //!   pointer on container
  T* mDataPtr{};                       //!   histo data
  T mXMin{};                           //!
  T mXMax{};                           //!
  T mBinSize{};                        //!
  T mBinSizeInv{};                     //!
  uint32_t mNBins{};                   //!

  ClassDefNV(FlatHisto1D, 2);
};

using FlatHisto1D_f = FlatHisto1D<float>;
using FlatHisto1D_d = FlatHisto1D<double>;

} // namespace dataformats
} // namespace o2

#endif
