// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <TH1F.h>
#include <type_traits>

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
         BinSizeInv,
         NServiceSlots };

  FlatHisto1D() = default;

  FlatHisto1D(int nb, T xmin, T xmax)
  {
    assert(nb > 0 && xmin < xmax);
    mData.resize(nb + NServiceSlots, 0.);
    mData[NBins] = nb;
    mData[XMin] = xmin;
    mData[XMax] = xmax;
    mData[BinSize] = (xmax - xmin) / nb;
    mData[BinSizeInv] = nb / (xmax - xmin);
    init();
  }

  FlatHisto1D(const gsl::span<const T> ext)
  {
    adoptExternal(ext);
  }

  int getNBins() const { return mNBins; }
  T getXMin() const { return mDataView[XMin]; }
  T getXMax() const { return mDataView[XMax]; }
  T getBinSize() const { return mDataView[BinSize]; }
  T getBinContent(uint32_t ib) const { return ib < mNBins ? mDataView[ib + NServiceSlots] : 0.; }
  T getBinContentForX(T x) const { getBinContent(getBin(x)); }

  T getBinStart(int i) const
  {
    assert(i < getNBins());
    return getXMin() + i * getBinSize();
  }

  T getBinCenter(int i) const
  {
    assert(i < getNBins());
    return getXMin() + (i + 0.5) * getBinSize();
  }

  T getBinEnd(int i) const
  {
    assert(i < getNBins());
    return getXMin() + (i + 1) * getBinSize();
  }

  void add(const FlatHisto1D& other)
  {
    assert(getNBins() == other.getNBins() && getXMin() == other.getXMin() && getXMax() == other.getXMax() && canFill());
    int last = NServiceSlots + getNBins();
    const auto& otherView = other.getView();
    for (int i = NServiceSlots; i < last; i++) {
      mData[i] += otherView[i];
    }
  }

  void subtract(const FlatHisto1D& other)
  {
    assert(getNBins() == other.getNBins() && getXMin() == other.getXMin() && getXMax() == other.getXMax() && canFill());
    int last = NServiceSlots + getNBins();
    const auto& otherView = other.getView();
    for (int i = NServiceSlots; i < last; i++) {
      mData[i] -= otherView[i];
    }
  }

  void setBinContent(uint32_t bin, T w)
  {
    assert(canFill() && bin < mNBins);
    mData[bin + NServiceSlots] = w;
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
    mNBins = (int)mDataView[NBins];
  }

  void init()
  { // when reading from file, need to call this method to make it operational
    assert(mData.size() > NServiceSlots);
    mDataView = gsl::span<const T>(mData.data(), mData.size());
    mNBins = (int)mData[NBins];
  }

  void fill(T x)
  {
    uint32_t bin = getBin(x);
    if (bin < mNBins) {
      mData[NServiceSlots + bin]++;
    }
  }

  void fill(T x, T w)
  {
    uint32_t bin = getBin(x);
    if (bin < mNBins) {
      mData[NServiceSlots + bin] += w;
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
    return mData.size() > NServiceSlots;
  }

  TH1F createTH1F(const std::string& name = "histo1d")
  {
    TH1F h(name.c_str(), name.c_str(), getNBins(), getXMin(), getXMax());
    for (int i = getNBins(); i--;) {
      auto w = getBinContent(i);
      if (w) {
        h.SetBinContent(i + 1, w);
      }
    }
    return h;
  }

  const std::vector<T>& getBase() const { return mData; }
  gsl::span<const T> getView() const { return mDataView; }

 protected:
  T getBinSizeInv() const { return mDataView[BinSizeInv]; }

  std::vector<T> mData;         // data to fill
  gsl::span<const T> mDataView; //!
  int mNBins = 0;               //!

  ClassDefNV(FlatHisto1D, 1);
};

using FlatHisto1D_f = FlatHisto1D<float>;
using FlatHisto1D_d = FlatHisto1D<double>;

} // namespace dataformats
} // namespace o2

#endif
