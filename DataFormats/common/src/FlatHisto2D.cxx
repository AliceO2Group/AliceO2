// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  FlatHisto1D.cxx
/// \brief 2D messeageable histo class
/// \author ruben.shahoyan@cern.ch

#include "CommonDataFormat/FlatHisto2D.h"
#include <TH1F.h>
#include <TH2F.h>

namespace o2
{
namespace dataformats
{

using namespace o2::dataformats;

template <typename T>
FlatHisto2D<T>::FlatHisto2D(int nbx, T xmin, T xmax, int nby, T ymin, T ymax)
{
  assert(nbx > 0 && xmin < xmax);
  assert(nby > 0 && ymin < ymax);
  mContainer.resize(nbx * nby + NServiceSlots, 0.);
  mContainer[NBinsX] = nbx;
  mContainer[NBinsY] = nby;
  mContainer[XMin] = xmin;
  mContainer[XMax] = xmax;
  mContainer[YMin] = ymin;
  mContainer[YMax] = ymax;
  mContainer[BinSizeX] = (xmax - xmin) / nbx;
  mContainer[BinSizeY] = (ymax - ymin) / nby;
  init(gsl::span<const T>(mContainer.data(), mContainer.size()));
}

template <typename T>
void FlatHisto2D<T>::adoptExternal(const gsl::span<const T> ext)
{
  assert(ext.size() > NServiceSlots);
  mContainer.clear();
  mContainerView = ext;
  init(mContainerView);
}

template <typename T>
void FlatHisto2D<T>::add(const FlatHisto2D& other)
{
  if (!(getNBinsX() == other.getNBinsX() && getXMin() == other.getXMin() && getXMax() == other.getXMax() &&
        getNBinsY() == other.getNBinsY() && getYMin() == other.getYMin() && getYMax() == other.getYMax() &&
        canFill())) {
    throw std::runtime_error("adding incompatible histos or destination histo is const");
  }
  for (int i = getNBins(); i--;) {
    mDataPtr[i] += other.mDataPtr[i];
  }
}

template <typename T>
void FlatHisto2D<T>::subtract(const FlatHisto2D& other)
{
  if (!(getNBinsX() == other.getNBinsX() && getXMin() == other.getXMin() && getXMax() == other.getXMax() &&
        getNBinsY() == other.getNBinsY() && getYMin() == other.getYMin() && getYMax() == other.getYMax() &&
        canFill())) {
    throw std::runtime_error("subtracting incompatible histos or destination histo is const");
  }
  for (int i = getNBins(); i--;) {
    mDataPtr[i] -= other.mDataPtr[i];
  }
}

template <typename T>
T FlatHisto2D<T>::getSum() const
{
  T sum = 0;
  for (int i = getNBins(); i--;) {
    sum += getBinContent(i);
  }
  return sum;
}

template <typename T>
void FlatHisto2D<T>::init(const gsl::span<const T> ext)
{ // when reading from file, need to call this method to make it operational
  assert(ext.size() > NServiceSlots);
  mContainerView = ext;
  mDataPtr = const_cast<T*>(&ext[NServiceSlots]);
  mNBinsX = (int)ext[NBinsX];
  mNBinsY = (int)ext[NBinsY];
  mXMin = ext[XMin];
  mXMax = ext[XMax];
  mYMin = ext[YMin];
  mYMax = ext[YMax];
  mBinSizeX = ext[BinSizeX];
  mBinSizeY = ext[BinSizeY];
  mBinSizeXInv = 1. / mBinSizeX;
  mBinSizeYInv = 1. / mBinSizeY;
}

template <typename T>
std::unique_ptr<TH2F> FlatHisto2D<T>::createTH2F(const std::string& name)
{
  auto h = std::make_unique<TH2F>(name.c_str(), name.c_str(), getNBinsX(), getXMin(), getXMax(), getNBinsY(), getYMin(), getYMax());
  for (int i = getNBinsX(); i--;) {
    for (int j = getNBinsY(); j--;) {
      auto w = getBinContent(i, j);
      if (w) {
        h->SetBinContent(i + 1, j + 1, w);
      }
    }
  }
  return std::move(h);
}

template <typename T>
std::unique_ptr<TH1F> FlatHisto2D<T>::createSliceYTH1F(uint32_t binX, const std::string& name) const
{
  auto h = std::make_unique<TH1F>(name.c_str(), name.c_str(), getNBinsY(), getYMin(), getYMax());
  if (binX < getNBinsX()) {
    for (int i = getNBinsY(); i--;) {
      h->SetBinContent(i + 1, getBinContent(binX, i));
    }
  }
  return h;
}

template <typename T>
std::unique_ptr<TH1F> FlatHisto2D<T>::createSliceXTH1F(uint32_t binY, const std::string& name) const
{
  auto h = std::make_unique<TH1F>(name.c_str(), name.c_str(), getNBinsX(), getXMin(), getXMax());
  if (binY < getNBinsY()) {
    for (int i = getNBinsX(); i--;) {
      h->SetBinContent(i + 1, getBinContent(i, binY));
    }
  }
  return h;
}

template class FlatHisto2D<float>;
template class FlatHisto2D<double>;

} // namespace dataformats
} // namespace o2