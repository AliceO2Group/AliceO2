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

  CalArray(const CalArray& calDet) :
    mName(calDet.mName),
    mData(calDet.mData)
  {}

  PadSubset getPadSubset() const { return mPadSubset; }
  int getPadSubsetNumber() const { return mPadSubsetNumber; }

  void setValue(const size_t channel, const T value) { mData[channel] = value; }
  const T& getValue(const size_t channel) const { return mData[channel]; }

  void setName(const std::string& name) { mName = name; }
  const std::string& getName() const { return mName; }

  const std::vector<T>& getData() const { return mData; }
  std::vector<T>& getData() { return mData; }

  const CalArray& operator+= (const CalArray& other);
private:
  std::string mName;
  // better to use std::array?
  //std::vector<T, Vc::Allocator<T>> mData;
  // how to use Vc::Allocator in this case? Specialisation for float, double, etc?
  std::vector<T> mData;       ///< calibration data
  PadSubset mPadSubset;       ///< Subset type
  int       mPadSubsetNumber; ///< Number of the pad subset, e.g. ROC 0 is IROC A00

  //ClassDefOverride(CalArray, 1);
};

// ===| pad region etc. initialisation |========================================
template <class T>
CalArray<T>::CalArray(PadSubset padSubset, int padSubsetNumber)
{
  const auto& mapper = Mapper::instance();
  mPadSubset       = padSubset;
  mPadSubsetNumber = padSubsetNumber;

  switch (padSubset) {
    case PadSubset::ROC: {
      mData.resize(ROC(padSubsetNumber).rocType() == RocType::IROC? mapper.getPadsInIROC() : mapper.getPadsInOROC());
      break;
    }
    case PadSubset::Partition: {
      mData.resize(mapper.getPartitionInfo(padSubsetNumber % mapper.getNumberOfPartitions()).getNumberOfPads());
      break;
    }
    case PadSubset::Region: {
      mData.resize(mapper.getPadRegionInfo(padSubsetNumber % mapper.getNumberOfPadRegions()).getNumberOfPads());
      break;
    }
  }
}

template <class T>
inline const CalArray<T>& CalArray<T>::operator+= (const CalArray& other)
{
  if ( !((mPadSubset == other.mPadSubset) && (mPadSubsetNumber == other.mPadSubsetNumber) ) ){
    return *this;
  }
}

using CalROC = CalArray<float>;

}
}

#endif
