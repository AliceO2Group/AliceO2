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

#include <Vc/Vc>
#include <memory>
#include <vector>
#include <string>
#include <type_traits>
#include <boost/format.hpp>

#include "TPCBase/Mapper.h"

using boost::format;

namespace o2 {
namespace TPC {

/// Class to hold calibration data on a pad level
/// 
/// Calibration data per pad for a certain subset of pads:
/// Full readout chamber, readout partition, or pad region
template <class T>
class CalArray {
public:
  /// Default constructor
  CalArray() = default;

  /// Default destructor
  ~CalArray() = default;

  CalArray(PadSubset padSubset, int padSubsetNumber);

  CalArray(const std::string name) :
    mName(name),
    mData()
  {}

  CalArray(const size_t size) :
    mName(),
    mData(size)
  {}

  //CalArray(const CalArray& calDet) :
    //mName(calDet.mName),
    //mData(calDet.mData),
    //mPadSubset(calDet.mPadSubset),
    //mPadSubsetNumber(calDet.mPadSubsetNumber)
  //{}

  //CalArray(CalArray&& calDet) :
    //mName(std::move(calDet.mName)),
    //mData(std::move(calDet.mData)),
    //mPadSubset(std::move(calDet.mPadSubset)),
    //mPadSubsetNumber(std::move(calDet.mPadSubsetNumber))
  //{}

  //CalArray& operator= (const CalArray& calArray)
  //{
    //return *this;
  //}

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

  const CalArray<T>& multiply(const T& val) { return *this *= val; }

  const CalArray& operator+= (const CalArray& other);
  const CalArray& operator*= (const T& val);
private:
  std::string mName;
  // better to use std::array?
  //std::vector<T, Vc::Allocator<T>> mData;
  // how to use Vc::Allocator in this case? Specialisation for float, double, etc?
  std::vector<T> mData;       ///< calibration data
  PadSubset mPadSubset;       ///< Subset type
  int       mPadSubsetNumber; ///< Number of the pad subset, e.g. ROC 0 is IROC A00
};

// ===| pad region etc. initialisation |========================================
template <class T>
CalArray<T>::CalArray(PadSubset padSubset, int padSubsetNumber)
  : mData(),
    mPadSubset(padSubset),
    mPadSubsetNumber(padSubsetNumber)
{
  const auto& mapper = Mapper::instance();

  switch (padSubset) {
    case PadSubset::ROC: {
      mData.resize(ROC(padSubsetNumber).rocType() == RocType::IROC? mapper.getPadsInIROC() : mapper.getPadsInOROC());
      setName(boost::str(format("ROC %1$02d") % padSubsetNumber));
      break;
    }
    case PadSubset::Partition: {
      mData.resize(mapper.getPartitionInfo(padSubsetNumber % mapper.getNumberOfPartitions()).getNumberOfPads());
      setName(boost::str(format("Partition %1$03d") % padSubsetNumber));
      break;
    }
    case PadSubset::Region: {
      mData.resize(mapper.getPadRegionInfo(padSubsetNumber % mapper.getNumberOfPadRegions()).getNumberOfPads());
      setName(boost::str(format("Region %1$03d") % padSubsetNumber));
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
inline const CalArray<T>& CalArray<T>::operator+= (const CalArray& other)
{
  if ( !((mPadSubset == other.mPadSubset) && (mPadSubsetNumber == other.mPadSubsetNumber) ) ){
    return *this;
  }
  for (size_t i=0; i<mData.size(); ++i) {
    mData[i] += other.getValue(i);
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalArray<T>& CalArray<T>::operator*= (const T& val)
{
  for (auto& data : mData) {
    data *= val;
  }
  return *this;
}

using CalROC = CalArray<float>;

}
}

#endif
