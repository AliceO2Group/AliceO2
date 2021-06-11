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
/// \brief 1D messeageable histo class
/// \author ruben.shahoyan@cern.ch

#include "CommonDataFormat/FlatHisto1D.h"
#include <TH1F.h>

namespace o2
{
namespace dataformats
{

template <typename T>
FlatHisto1D<T>::FlatHisto1D(int nb, T xmin, T xmax)
{
  assert(nb > 0 && xmin < xmax);
  mContainer.resize(nb + NServiceSlots, 0.);
  mContainer[NBins] = nb;
  mContainer[XMin] = xmin;
  mContainer[XMax] = xmax;
  mContainer[BinSize] = (xmax - xmin) / nb;
  init(gsl::span<const T>(mContainer.data(), mContainer.size()));
}

template <typename T>
void FlatHisto1D<T>::adoptExternal(const gsl::span<const T> ext)
{
  assert(ext.size() > NServiceSlots);
  mContainer.clear();
  mContainerView = ext;
  init(mContainerView);
}

template <typename T>
void FlatHisto1D<T>::add(const FlatHisto1D& other)
{
  if (!(getNBins() == other.getNBins() && getXMin() == other.getXMin() && getXMax() == other.getXMax() && canFill())) {
    throw std::runtime_error("adding incompatible histos or destination histo is const");
  }
  for (int i = getNBins(); i--;) {
    mDataPtr[i] += other.mDataPtr[i];
  }
}

template <typename T>
void FlatHisto1D<T>::subtract(const FlatHisto1D& other)
{
  if (!(getNBins() == other.getNBins() && getXMin() == other.getXMin() && getXMax() == other.getXMax() && canFill())) {
    throw std::runtime_error("subtracting incompatible histos or destination histo is const");
  }
  for (int i = getNBins(); i--;) {
    mDataPtr[i] -= other.mDataPtr[i];
  }
}

template <typename T>
T FlatHisto1D<T>::getSum() const
{
  T sum = 0;
  for (int i = getNBins(); i--;) {
    sum += getBinContent(i);
  }
  return sum;
}

template <typename T>
void FlatHisto1D<T>::init(const gsl::span<const T> ext)
{ // when reading from file, need to call this method to make it operational
  assert(ext.size() > NServiceSlots);
  mContainerView = ext;
  mDataPtr = const_cast<T*>(&ext[NServiceSlots]);
  mNBins = (int)ext[NBins];
  mXMin = ext[XMin];
  mXMax = ext[XMax];
  mBinSize = ext[BinSize];
  mBinSizeInv = 1. / mBinSize;
}

template <typename T>
std::unique_ptr<TH1F> FlatHisto1D<T>::createTH1F(const std::string& name)
{
  auto h = std::make_unique<TH1F>(name.c_str(), name.c_str(), getNBins(), getXMin(), getXMax());
  for (int i = getNBins(); i--;) {
    auto w = getBinContent(i);
    if (w) {
      h->SetBinContent(i + 1, w);
    }
  }
  return std::move(h);
}

template class FlatHisto1D<float>;
template class FlatHisto1D<double>;

} // namespace dataformats
} // namespace o2