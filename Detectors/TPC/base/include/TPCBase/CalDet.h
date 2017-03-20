#ifndef ALICEO2_TPC_CALDET_H_
#define ALICEO2_TPC_CALDET_H_

#include <Vc/Vc>
#include <memory>
#include <vector>
#include <string>

#include "TPCBase/Mapper.h"
#include "TPCBase/ROC.h"
#include "TPCBase/CalArray.h"

namespace AliceO2 {
namespace TPC {

/// Class to hold calibration data on a pad level
///
template <class T>
class CalDet {
  using CalType = CalArray<T>;
//using T = float;
public:
  enum class PadSubset : char {
    ROC,        ///< ROCs (up to 72)
    Partition,  ///< Partitions (up to 36*5)
    Region      ///< Regions (up to 36*10)
  };

  CalDet() {};
  CalDet(PadSubset padSubset);
  CalDet(const std::string name) : 
    mName(name),
    mData()
  {}

  CalDet(const CalDet& calDet) :
    mName(calDet.mName),
    mData(calDet.mData)
  {}

  const std::vector<CalType>& getData() const { return mData; }

  //void setValue(const unsigned int channel, const T value) { mData[channel] = value; }
  //const T& getValue(const unsigned int channel) const { return mData[channel]; }

  CalType& getCalArray(size_t position) const { return mData[position]; }

  void setName(const std::string& name) { mName = name; }
  const std::string& getName() const { return mName; }

  const CalDet& operator+= (const CalDet& other);
private:
  std::string mName;
  // better to use std::array?
  std::vector<CalType> mData;
  PadSubset mPadSubset; ///< Pad subset granularity
};


template <class T>
inline const CalDet<T>& CalDet<T>::operator+= (const CalDet& other)
{
}

// ===| Full detector initialisation |==========================================
template <class T>
CalDet<T>::CalDet(PadSubset padSusbset)
{
  const auto& mapper = Mapper::instance();

  format name;
  // ---| Define number of sub pad regions |------------------------------------
  size_t size;
  typename CalType::PadSubset subsetType;

  switch (padSusbset) {
    case PadSubset::ROC: {
      size = ROC::MaxROC;
      subsetType = CalType::PadSubset::ROC;
      break;
    }
    case PadSubset::Partition: {
      size = Sector::MaxSector * mapper.getNumberOfPartitions();
      subsetType = CalType::PadSubset::Partition;
      break;
    }
    case PadSubset::Region: {
      size = Sector::MaxSector * mapper.getNumberOfPadRegions();
      subsetType = CalType::PadSubset::Region;
      break;
    }
  }

  for (size_t i=0; i<size; ++i) {
    mData.push_back(CalType(subsetType, i));
    //mData.push_back(CalArray<T>(CalArray<T>::PadSubset(char(padSusbset)), i));
    //mData.push_back(CalType(padSusbset, i));
  }
}

using CalPad = CalDet<float>;

}
}

#endif
