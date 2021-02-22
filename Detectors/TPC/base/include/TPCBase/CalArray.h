// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALARRAY_H_
#define ALICEO2_TPC_CALARRAY_H_

#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <type_traits>

#include "TPCBase/Mapper.h"

#ifndef GPUCA_ALIGPUCODE
#include "FairLogger.h"
#include <boost/format.hpp>
#endif

namespace o2
{
namespace tpc
{

/// Class to hold calibration data on a pad level
///
/// Calibration data per pad for a certain subset of pads:
/// Full readout chamber, readout partition, or pad region
template <class T>
class CalArray
{
 public:
  /// Default constructor
  CalArray() = default;

  /// Default destructor
  ~CalArray() = default;

  /// Constructor sets a default name depending on the pad subset type and number
  /// \param padSubset pad subset type (e.g. PadSubset::ROC)
  /// \param padSubsetNumber number of the pad subset (e.g. 0 for ROC 0)
  CalArray(const PadSubset padSubset, const int padSubsetNumber)
    : mName(),
      mData(),
      mPadSubset(padSubset),
      mPadSubsetNumber(padSubsetNumber)
  {
    // initialize the data array
    initData();
  }

  /// Constructor assumes PadSubset::ROC
  /// \param name name of the calibration array
  /// \param padSubsetNumber number of the pad subset (e.g. 0 for ROC 0)
  CalArray(const std::string_view name, const int padSubsetNumber)
    : mName(name),
      mData(),
      mPadSubset(PadSubset::ROC),
      mPadSubsetNumber(padSubsetNumber)
  {
    // initialize the data array
    initData();
  }

  /// Constructor
  /// \param name name of the calibration array
  /// \param padSubsetNumber number of the pad subset (e.g. 0 for ROC 0)
  CalArray(const std::string_view name, const PadSubset padSubset, const int padSubsetNumber)
    : mName(name),
      mData(),
      mPadSubset(padSubset),
      mPadSubsetNumber(padSubsetNumber)
  {
    // initialize the data array
    initData();
  }

  /// Return the pad subset type
  /// \return pad subset type
  PadSubset getPadSubset() const { return mPadSubset; }

  /// Return the pad subset number (e.g. ROC number)
  /// \return pad subset number (e.g. ROC number)
  int getPadSubsetNumber() const { return mPadSubsetNumber; }

  void setValue(const size_t channel, const T& value) { mData[channel] = value; }
  const T getValue(const size_t channel) const { return mData[channel]; }

  void setValue(const size_t row, const size_t pad, const T& value);
  const T getValue(const size_t row, const size_t pad) const;

  void setName(const std::string& name) { mName = name; }
  const std::string& getName() const { return mName; }

  const std::vector<T>& getData() const { return mData; }
  std::vector<T>& getData() { return mData; }

  /// calculate the sum of all elements
  const T getSum() const { return std::accumulate(mData.begin(), mData.end(), T(0)); }

  /// Multiply all val to all channels
  const CalArray<T>& multiply(const T& val) { return *this *= val; }

  /// Add other to this channel by channel
  const CalArray& operator+=(const CalArray& other);

  /// Subtract other from this channel by channel
  const CalArray& operator-=(const CalArray& other);

  /// Multiply other to this channel by channel
  const CalArray& operator*=(const CalArray& other);

  /// Divide this by other channel by channel
  const CalArray& operator/=(const CalArray& other);

  /// Add value to all channels
  const CalArray& operator+=(const T& val);

  /// Subtract value from all channels
  const CalArray& operator-=(const T& val);

  /// Multiply value to all channels
  const CalArray& operator*=(const T& val);

  /// Divide value on all channels
  const CalArray& operator/=(const T& val);

 private:
  std::string mName;
  // better to use std::array?
  //std::vector<T, Vc::Allocator<T>> mData;
  // how to use Vc::Allocator in this case? Specialisation for float, double, etc?
  std::vector<T> mData; ///< calibration data
  PadSubset mPadSubset; ///< Subset type
  int mPadSubsetNumber; ///< Number of the pad subset, e.g. ROC 0 is IROC A00

  /// initialize the data array depending on what is set as PadSubset
  void initData();
};

#ifndef GPUCA_ALIGPUCODE

// ===| pad region etc. initialisation |========================================
template <class T>
void CalArray<T>::initData()
{
  const auto& mapper = Mapper::instance();

  switch (mPadSubset) {
    case PadSubset::ROC: {
      mData.resize(ROC(mPadSubsetNumber).rocType() == RocType::IROC ? mapper.getPadsInIROC() : mapper.getPadsInOROC());
      if (mName.empty()) {
        setName(boost::str(boost::format("ROC_%1$02d") % mPadSubsetNumber));
      }
      break;
    }
    case PadSubset::Partition: {
      mData.resize(mapper.getPartitionInfo(mPadSubsetNumber % mapper.getNumberOfPartitions()).getNumberOfPads());
      if (mName.empty()) {
        setName(boost::str(boost::format("Partition_%1$03d") % mPadSubsetNumber));
      }
      break;
    }
    case PadSubset::Region: {
      mData.resize(mapper.getPadRegionInfo(mPadSubsetNumber % mapper.getNumberOfPadRegions()).getNumberOfPads());
      if (mName.empty()) {
        setName(boost::str(boost::format("Region_%1$03d") % mPadSubsetNumber));
      }
      break;
    }
  }
}
//______________________________________________________________________________
template <class T>
inline void CalArray<T>::setValue(const size_t row, const size_t pad, const T& value)
{
  /// \todo might need check for row, pad or position limits
  static const auto& mapper = Mapper::instance();
  size_t position = mapper.getPadNumber(mPadSubset, mPadSubsetNumber, row, pad);
  setValue(position, value);
}

//______________________________________________________________________________
template <class T>
inline const T CalArray<T>::getValue(const size_t row, const size_t pad) const
{
  /// \todo might need check for row, pad or position limits
  static const auto& mapper = Mapper::instance();
  size_t position = mapper.getPadNumber(mPadSubset, mPadSubsetNumber, row, pad);
  return getValue(position);
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator+=(const CalArray<T>& other)
{
  if (!((mPadSubset == other.mPadSubset) && (mPadSubsetNumber == other.mPadSubsetNumber))) {
    LOG(ERROR) << "You are trying to operate on incompatible objects: Pad subset type and number must be the same on both objects";
    return *this;
  }
  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] += other.getValue(i);
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator-=(const CalArray<T>& other)
{
  if (!((mPadSubset == other.mPadSubset) && (mPadSubsetNumber == other.mPadSubsetNumber))) {
    LOG(ERROR) << "You are trying to operate on incompatible objects: Pad subset type and number must be the same on both objects";
    return *this;
  }
  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] -= other.getValue(i);
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator*=(const CalArray<T>& other)
{
  if (!((mPadSubset == other.mPadSubset) && (mPadSubsetNumber == other.mPadSubsetNumber))) {
    LOG(ERROR) << "pad subset type of the objects it not compatible";
    return *this;
  }
  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] *= other.getValue(i);
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator/=(const CalArray<T>& other)
{
  if (!((mPadSubset == other.mPadSubset) && (mPadSubsetNumber == other.mPadSubsetNumber))) {
    LOG(ERROR) << "pad subset type of the objects it not compatible";
    return *this;
  }
  for (size_t i = 0; i < mData.size(); ++i) {
    if (other.getValue(i) != 0) {
      mData[i] /= other.getValue(i);
    } else {
      mData[i] = 0;
      LOG(ERROR) << "Division by 0 detected! Value was set to 0.";
    }
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator+=(const T& val)
{
  for (auto& data : mData) {
    data += val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator-=(const T& val)
{
  for (auto& data : mData) {
    data -= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator*=(const T& val)
{
  for (auto& data : mData) {
    data *= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator/=(const T& val)
{
  for (auto& data : mData) {
    data /= val;
  }
  return *this;
}

using CalROC = CalArray<float>;

#endif // GPUCA_ALIGPUCODE

} // namespace tpc
} // namespace o2

#endif
