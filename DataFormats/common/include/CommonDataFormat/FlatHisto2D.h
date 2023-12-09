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

/// @file  FlatHisto2D.h
/// \brief 2D messeageable histo class
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FLATHISTO2D_H
#define ALICEO2_FLATHISTO2D_H

#include <Rtypes.h>
#include <vector>
#include <gsl/span>
#include <type_traits>
#include <cassert>
#include <memory>

class TH1F;
class TH2F;

namespace o2
{
namespace dataformats
{

/*
  Fast 2D histo class which can be messages as
  FlatHisto2D<float> histo(nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  histo.fill(...);
  pc.outputs().snapshot(Output{"Origin", "Desc", 0}, histo.getBase());

  and received (read only!) as
  const auto hdata = pc.inputs().get<gsl::span<float>>("histodata");
  FlatHisto2D<float> histoView;
  histoView.adoptExternal(hdata);
  or directly
  FlatHisto2D<float> histoView(pc.inputs().get<gsl::span<float>>("histodata"));
*/

template <typename T = float>
class FlatHisto2D
{
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "T must be float or double");

 public:
  enum { NBinsX,
         NBinsY,
         XMin,
         XMax,
         YMin,
         YMax,
         BinSizeX,
         BinSizeY,
         NServiceSlots };

  FlatHisto2D() = default;
  FlatHisto2D(uint32_t nbx, T xmin, T xmax, uint32_t nby, T ymin, T ymax);
  FlatHisto2D(const gsl::span<const T> ext) { adoptExternal(ext); }
  FlatHisto2D(const FlatHisto2D& src);
  FlatHisto2D& operator=(const FlatHisto2D& rhs);
  void adoptExternal(const gsl::span<const T> ext);
  void init()
  {
    // when reading from file, need to call this method to make it operational
    assert(mContainer.size() > NServiceSlots);
    init(gsl::span<const T>(mContainer.data(), mContainer.size()));
  }
  void init(uint32_t nbx, T xmin, T xmax, uint32_t nby, T ymin, T ymax);
  uint32_t getNBinsX() const { return mNBinsX; }
  uint32_t getNBinsY() const { return mNBinsY; }
  uint32_t getNBins() const { return getNBinsX() * getNBinsY(); }

  T getXMin() const { return mXMin; }
  T getXMax() const { return mXMax; }
  T getYMin() const { return mYMin; }
  T getYMax() const { return mYMax; }
  T getBinSizeX() const { return mBinSizeX; }
  T getBinSizeY() const { return mBinSizeY; }
  T getBinSizeXInv() const { return mBinSizeXInv; }
  T getBinSizeYInv() const { return mBinSizeYInv; }

  T getBinContent(uint32_t ib) const
  {
    assert(ib < getNBins());
    return mDataPtr[ib];
  }

  T getBinContent(uint32_t ibx, uint32_t iby) const { return getBinContent(getGlobalBin(ibx, iby)); }

  T getBinContentForXY(T x, T y) const { return getBinContent(getBinX(x), getBinY(y)); }

  bool isValidBin(uint32_t bin) const { return bin < getNBins(); }
  bool isBinEmpty(uint32_t bin) const { return getBinContent(bin) == 0; }

  T getBinXStart(uint32_t i) const
  {
    assert(i < getNBinsX());
    return getXMin() + i * getBinSizeX();
  }

  T getBinXCenter(uint32_t i) const
  {
    assert(i < getNBinsX());
    return getXMin() + (i + 0.5) * getBinSizeX();
  }

  T getBinXEnd(uint32_t i) const
  {
    assert(i < getNBinsX());
    return getXMin() + (i + 1) * getBinSizeX();
  }

  T getBinYStart(uint32_t i) const
  {
    assert(i < getNBinsY());
    return getYMin() + i * getBinSizeY();
  }

  T getBinYCenter(uint32_t i) const
  {
    assert(i < getNBinsY());
    return getYMin() + (i + 0.5) * getBinSizeY();
  }

  T getBinYEnd(uint32_t i) const
  {
    assert(i < getNBinsY());
    return getYMin() + (i + 1) * getBinSizeY();
  }

  uint32_t getXBin(uint32_t i) const { return i / getNBinsY(); }
  uint32_t getYBin(uint32_t i) const { return i % getNBinsY(); }

  void add(const FlatHisto2D& other);

  void subtract(const FlatHisto2D& other);

  void setBinContent(uint32_t bin, T w)
  {
    assert(canFill() && isValidBin(bin));
    mDataPtr[bin] = w;
  }

  void setBinContent(uint32_t binX, uint32_t binY, T w)
  {
    auto bin = getGlobalBin(binX, binY);
    setBinContent(bin, w);
  }

  void clear()
  {
    assert(canFill());
    memset(mDataPtr, 0, sizeof(T) * getNBins());
  }

  T getSum() const;

  int fill(T x, T y)
  {
    uint32_t bin = getBin(x, y);
    if (isValidBin(bin)) {
      mDataPtr[bin]++;
      return (int)bin;
    }
    return -1;
  }

  int fill(T x, T y, T w)
  {
    uint32_t bin = getBin(x, y);
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

  void fillBin(uint32_t bx, uint32_t by, T w)
  {
    auto bin = getGlobalBin(bx, by);
    if (isValidBin(bin)) {
      mDataPtr[bin] += w;
    }
  }

  uint32_t getBinX(T x) const
  {
    auto dx = x - getXMin();
    return dx < 0 ? 0xffffffff : uint32_t(dx * getBinSizeXInv());
  }

  uint32_t getBinY(T y) const
  {
    auto dy = y - getYMin();
    return dy < 0 ? 0xffffffff : uint32_t(dy * getBinSizeYInv());
  }

  uint32_t getBin(T x, T y) const
  {
    auto bx = getBinX(x), by = getBinY(y);
    return bx < getNBinsX() && by < getNBinsY() ? getGlobalBin(bx, by) : 0xffffffff;
  }

  bool canFill() const
  {
    // histo can be filled only if hase its own data, otherwise only query can be done on the view
    return mContainer.size() > NServiceSlots;
  }

  gsl::span<const T> getSliceY(uint32_t binX) const
  {
    uint32_t offs = binX * getNBinsY();
    return binX < getNBinsX() ? gsl::span<const T>(&mDataPtr[offs], getNBinsY()) : gsl::span<const T>();
  }

  std::unique_ptr<TH2F> createTH2F(const std::string& name = "histo2d") const;

  std::unique_ptr<TH1F> createSliceXTH1F(uint32_t binY, const std::string& name = "histo2dsliceX") const;
  std::unique_ptr<TH1F> createSliceYTH1F(uint32_t binX, const std::string& name = "histo2dsliceY") const;

  const std::vector<T>& getBase() const { return mContainer; }
  gsl::span<const T> getView() const { return mContainerView; }

  uint32_t getGlobalBin(uint32_t binX, uint32_t binY) const { return binX * getNBinsY() + binY; }

 protected:
  void init(const gsl::span<const T> ext);

  std::vector<T> mContainer;         // data to fill
  gsl::span<const T> mContainerView; //!
  T* mDataPtr{};                     //!   histo data
  T mXMin{};                         //!
  T mXMax{};                         //!
  T mYMin{};                         //!
  T mYMax{};                         //!
  T mBinSizeX{};                     //!
  T mBinSizeY{};                     //!
  T mBinSizeXInv{};                  //!
  T mBinSizeYInv{};                  //!
  uint32_t mNBinsX{};                //!
  uint32_t mNBinsY{};                //!
  uint32_t mNBins{};                 //!

  ClassDefNV(FlatHisto2D, 2);
};

using FlatHisto2D_f = FlatHisto2D<float>;
using FlatHisto2D_d = FlatHisto2D<double>;

} // namespace dataformats
} // namespace o2

#endif
