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

#ifndef O2_MCH_CALIBRATION_PEDESTAL_DATA_H_
#define O2_MCH_CALIBRATION_PEDESTAL_DATA_H_

#include "MCHCalibration/PedestalChannel.h"
#include "MCHCalibration/PedestalDigit.h"
#include "Rtypes.h"
#include <array>
#include <gsl/span>
#include <unordered_map>

namespace o2::mch::calibration
{

class PedestalDigit;
class PedestalData;

namespace impl
{
template <typename T>
class PedestalDataIterator;
}

/**
 * @class PedestalData 
 * @brief Compute and store the mean and RMS of the pedestal digit amplitudes
 *
 * To extract the values from PedestalData, use the provided iterator(s).
 *
 * @example
 * PedestalData data;
 * data.fill(...);
 * for (const auto& p: data) { std::cout << p << "\n"; }
 *
 */
class PedestalData
{
 public:
  using iterator = impl::PedestalDataIterator<PedestalChannel>;
  using const_iterator = impl::PedestalDataIterator<const PedestalChannel>;

  friend iterator;
  friend const_iterator;

  iterator begin();
  iterator end();
  const_iterator cbegin() const;
  const_iterator cend() const;

  PedestalData() = default;
  ~PedestalData() = default;

  [[deprecated("use fill method instead")]] void process(gsl::span<const PedestalDigit> digits);
  void reset();

  static constexpr int MAXDS = 40;      // max number of dual sampas per solar
  static constexpr int MAXCHANNEL = 64; // max number of channels per dual sampa

  /** a matrix of 40 (dual sampas) x 64 (channels) PedestalChannel objects */
  using PedestalMatrix = std::array<std::array<PedestalChannel, MAXCHANNEL>, MAXDS>;
  /** a map from solarIds to PedestalMatrix */
  using PedestalsMap = std::unordered_map<int, PedestalMatrix>;

  /** function to update the pedestal values from the data 
   * @param digits a span of pedestal digits for a single TimeFrame
  */
  void fill(const gsl::span<const PedestalDigit> digits);

  /** merge this object with other
  * FIXME: not yet implemented.
  */
  void merge(const PedestalData* other);

  /** dump this object. */
  void print() const;

  [[deprecated]] PedestalsMap getPedestals() const { return mPedestals; }

 private:
  PedestalsMap mPedestals{}; ///< internal storage of all PedestalChannel values

  ClassDefNV(PedestalData, 1)
};

namespace impl
{
template <typename T>
class PedestalDataIterator
{
 public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::forward_iterator_tag;

  PedestalDataIterator() = default;
  explicit PedestalDataIterator(PedestalData* data) : mData{data},
                                                      mMapIt{},
                                                      mCol{0},
                                                      mRow{0}
  {
    if (mData) {
      mMapIt = mData->mPedestals.begin();
    }
  }

  reference operator*() const
  {
    return mMapIt->second[mRow][mCol];
  }
  pointer operator->() const
  {
    return &mMapIt->second[mRow][mCol];
  }

  PedestalDataIterator& operator++()
  {
    if (mMapIt != mData->mPedestals.end()) {
      if (mCol < PedestalData::MAXCHANNEL - 1) {
        mCol++;
        return *this;
      }
      if (mRow < PedestalData::MAXDS - 1) {
        mCol = 0;
        mRow++;
        return *this;
      }
      ++mMapIt;
      mCol = 0;
      mRow = 0;
      if (mMapIt != mData->mPedestals.end()) {
        return *this;
      }
      mData = nullptr;
    }
    // undefined behavior here (should not increment an end iterator)
    return *this;
  }

  bool operator==(const PedestalDataIterator& rhs)
  {
    return mData == rhs.mData &&
           mMapIt == rhs.mMapIt &&
           mCol == rhs.mCol &&
           mRow == rhs.mRow;
  }

  bool operator!=(const PedestalDataIterator& rhs) { return !(*this == rhs); }

  PedestalDataIterator operator++(int)
  {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

 private:
  PedestalData* mData;
  PedestalData::PedestalsMap::iterator mMapIt;
  int mCol;
  int mRow;
};
} // namespace impl

} // namespace o2::mch::calibration
#endif
