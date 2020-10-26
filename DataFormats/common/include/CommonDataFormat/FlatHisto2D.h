// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <TH2F.h>
#include <type_traits>

namespace o2
{
namespace dataformats
{

/* 
  Fast 2D histo class which can be messages as
  FlatHisto2D<float> histo(nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  histo.fill(...);
  pc.outputs().snapshot(Output{"Origin", "Desc", 0, Lifetime::Timeframe}, histo.getBase());
  
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
         BinSizeXInv,
         BinSizeYInv,
         NServiceSlots };

  FlatHisto2D() = default;

  FlatHisto2D(int nbx, T xmin, T xmax, int nby, T ymin, T ymax)
  {
    assert(nbx > 0 && xmin < xmax);
    assert(nby > 0 && ymin < ymax);
    mData.resize(nbx * nby + NServiceSlots, 0.);
    mData[NBinsX] = nbx;
    mData[NBinsY] = nby;
    mData[XMin] = xmin;
    mData[XMax] = xmax;
    mData[YMin] = ymin;
    mData[YMax] = ymax;
    mData[BinSizeX] = (xmax - xmin) / nbx;
    mData[BinSizeXInv] = nbx / (xmax - xmin);
    mData[BinSizeY] = (ymax - ymin) / nby;
    mData[BinSizeYInv] = nby / (ymax - ymin);
    init();
  }

  FlatHisto2D(const gsl::span<const T> ext)
  {
    adoptExternal(ext);
  }

  int getNBinsX() const { return mNBinsX; }
  int getNBinsY() const { return mNBinsY; }
  int getNBins() const { return mNBins; }
  T getXMin() const { return mDataView[XMin]; }
  T getXMax() const { return mDataView[XMax]; }
  T getYMin() const { return mDataView[YMin]; }
  T getYMax() const { return mDataView[YMax]; }
  T getBinSizeX() const { return mDataView[BinSizeX]; }
  T getBinSizeY() const { return mDataView[BinSizeY]; }
  T getBinContent(uint32_t ib) const { return ib < mNBins ? mDataView[ib + NServiceSlots] : 0.; }
  T getBinContent(uint32_t ibx, uint32_t iby) const { return getBinContent(getGlobalBin(ibx, iby)); }
  T getBinContentForXY(T x, T y) const { return getBinContent(getBinX(x), getBinY(y)); }

  T getBinStartX(int i) const
  {
    assert(i < getNBinsX());
    return getXMin() + i * getBinSizeX();
  }

  T getBinCenterX(int i) const
  {
    assert(i < getNBinsX());
    return getXMin() + (i + 0.5) * getBinSizeX();
  }

  T getBinEndX(int i) const
  {
    assert(i < getNBinsX());
    return getXMin() + (i + 1) * getBinSizeX();
  }

  T getBinStartY(int i) const
  {
    assert(i < getNBinsY());
    return getYMin() + i * getBinSizeY();
  }

  T getBinCenterY(int i) const
  {
    assert(i < getNBinsY());
    return getYMin() + (i + 0.5) * getBinSizeY();
  }

  T getBinEndY(int i) const
  {
    assert(i < getNBinsY());
    return getYMin() + (i + 1) * getBinSizeY();
  }

  void add(const FlatHisto2D& other)
  {
    assert(getNBinsX() == other.getNBinsX() && getXMin() == other.getXMin() && getXMax() == other.getXMax() &&
           getNBinsY() == other.getNBinsY() && getYMin() == other.getYMin() && getYMax() == other.getYMax() &&
           canFill());
    int last = NServiceSlots + getNBins();
    const auto& otherView = other.getView();
    for (int i = NServiceSlots; i < last; i++) {
      mData[i] += otherView[i];
    }
  }

  void subtract(const FlatHisto2D& other)
  {
    assert(getNBinsX() == other.getNBinsX() && getXMin() == other.getXMin() && getXMax() == other.getXMax() &&
           getNBinsY() == other.getNBinsY() && getYMin() == other.getYMin() && getYMax() == other.getYMax() &&
           canFill());
    int last = NServiceSlots + getNBins();
    const auto& otherView = other.getView();
    for (int i = NServiceSlots; i < last; i++) {
      mData[i] -= otherView[i];
    }
  }

  void setBinContent(uint32_t bin, T w)
  {
    assert(canFill());
    if (bin < getNBins()) {
      mData[bin + NServiceSlots] = w;
    }
  }

  void setBinContent(uint32_t binX, uint32_t binY, T w)
  {
    assert(canFill());
    auto bin = getGlobalBin(binX, binY);
    if (bin < getNBins()) {
      mData[+NServiceSlots] = w;
    }
  }

  void clear()
  {
    assert(canFill());
    memset(mData.data() + NServiceSlots, 0, sizeof(T) * getNBins());
  }

  T getSum() const
  {
    T sum = 0;
    for (int i = getNBins(); i--;) {
      sum += getBinContent(i);
    }
    return sum;
  }

  void adoptExternal(const gsl::span<const T> ext)
  {
    assert(ext.size() > NServiceSlots);
    mData.clear();
    mDataView = ext;
    mNBinsX = (int)mDataView[NBinsX];
    mNBinsY = (int)mDataView[NBinsY];
    mNBins = mNBinsX * mNBinsY;
  }

  void init()
  { // when reading from file, need to call this method to make it operational
    assert(mData.size() > NServiceSlots);
    mDataView = gsl::span<const T>(mData.data(), mData.size());
    mNBinsX = (int)mData[NBinsX];
    mNBinsY = (int)mData[NBinsY];
    mNBins = mNBinsX * mNBinsY;
  }

  void fill(T x, T y)
  {
    uint32_t bin = getBin(x, y);
    if (bin < mNBins) {
      mData[NServiceSlots + bin]++;
    }
  }

  void fill(T x, T y, T w)
  {
    uint32_t bin = getBin(x, y);
    if (bin < mNBins) {
      mData[NServiceSlots + bin] += w;
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
    return mData.size() > NServiceSlots;
  }

  gsl::span<const T> getSliceY(uint32_t binX) const
  {
    int offs = NServiceSlots + binX * getNBinsY();
    return binX < getNBinsX() ? gsl::span<const T>(&mDataView[offs], getNBinsY()) : gsl::span<const T>();
  }

  TH1F createSliceYTH1F(uint32_t binX, const std::string& name = "histo2dslice") const
  {
    TH1F h(name.c_str(), name.c_str(), getNBinsY(), getYMin(), getYMax());
    if (binX < getNBinsX()) {
      for (int i = getNBinsY(); i--;) {
        h.SetBinContent(i + 1, getBinContent(binX, i));
      }
    }
    return h;
  }

  TH2F createTH2F(const std::string& name = "histo2d")
  {
    TH2F h(name.c_str(), name.c_str(), getNBinsX(), getXMin(), getXMax(), getNBinsY(), getYMin(), getYMax());
    for (int i = getNBinsX(); i--;) {
      for (int j = getNBinsY(); j--;) {
        auto w = getBinContent(i, j);
        if (w) {
          h.SetBinContent(i + 1, j + 1, w);
        }
      }
    }
    return h;
  }

  const std::vector<T>& getBase() const { return mData; }
  gsl::span<const T> getView() const { return mDataView; }

  int getGlobalBin(uint32_t binX, uint32_t binY) const { return binX * getNBinsY() + binY; }

 protected:
  T getBinSizeXInv() const { return mDataView[BinSizeXInv]; }
  T getBinSizeYInv() const { return mDataView[BinSizeYInv]; }

  std::vector<T> mData;         // data to fill
  gsl::span<const T> mDataView; //!
  int mNBinsX = 0;              //!
  int mNBinsY = 0;              //!
  int mNBins = 0;               //!

  ClassDefNV(FlatHisto2D, 1);
};

using FlatHisto2D_f = FlatHisto2D<float>;
using FlatHisto2D_d = FlatHisto2D<double>;

} // namespace dataformats
} // namespace o2

#endif
