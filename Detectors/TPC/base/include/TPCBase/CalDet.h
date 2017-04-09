#ifndef ALICEO2_TPC_CALDET_H_
#define ALICEO2_TPC_CALDET_H_

#include <Vc/Vc>
#include <memory>
#include <vector>
#include <string>

#include "TPCBase/Defs.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/ROC.h"
#include "TPCBase/CalArray.h"

namespace o2 {
namespace TPC {

/// Class to hold calibration data on a pad level
///
template <class T>
class CalDet {
  using CalType = CalArray<T>;
public:
  CalDet() = default;
  ~CalDet() = default;

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
  std::vector<CalType>& getData() { return mData; }

  //void setValue(const unsigned int channel, const T value) { mData[channel] = value; }
  //const T& getValue(const unsigned int channel) const { return mData[channel]; }

  const CalType& getCalArray(size_t position) const { return mData[position]; }
  CalType& getCalArray(size_t position) { return mData[position]; }

  void setName(const std::string& name) { mName = name; }
  const std::string& getName() const { return mName; }

  const CalDet& operator+= (const CalDet& other);
private:
  std::string mName;          ///< name of the object
  std::vector<CalType> mData; ///< internal CalArrays
  PadSubset mPadSubset;       ///< Pad subset granularity
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
  PadSubset subsetType;

  switch (padSusbset) {
    case PadSubset::ROC: {
      size = ROC::MaxROC;
      subsetType = PadSubset::ROC;
      break;
    }
    case PadSubset::Partition: {
      size = Sector::MaxSector * mapper.getNumberOfPartitions();
      subsetType = PadSubset::Partition;
      break;
    }
    case PadSubset::Region: {
      size = Sector::MaxSector * mapper.getNumberOfPadRegions();
      subsetType = PadSubset::Region;
      break;
    }
  }

  for (size_t i=0; i<size; ++i) {
    mData.push_back(CalType(subsetType, i));
  }
}

using CalPad = CalDet<float>;

}
}

#endif
