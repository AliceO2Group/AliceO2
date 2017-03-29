#ifndef ALICEO2_TPC_CALDET_H_
#define ALICEO2_TPC_CALDET_H_

#include <Vc/Vc>
#include <memory>
#include <vector>
#include <string>

#include <TPCBase/Mapper.h>

namespace AliceO2 {
namespace TPC {

/// Class to hold calibration data on a pad level
///
/// Might be removed again
template <class T>
class CalDet {
//using T = float;
public:
  CalDet() {};
  CalDet(const std::string name) : 
    mName(name),
    mData(Mapper::getPadsInSector()*36)
  {}

  CalDet(const CalDet& calDet) :
    mName(calDet.mName),
    mData(calDet.mData)
  {}

  void setValue(const unsigned int channel, const T value) { mData[channel] = value; }
  const T& getValue(const unsigned int channel) const { return mData[channel]; }

  void setName(const std::string& name) { mName = name; }
  const std::string& getName() const { return mName; }

  const CalDet& operator+= (const CalDet& other);
private:
  std::string mName;
  // better to use std::array?
  std::vector<T, Vc::Allocator<T>> mData;
};


template <class T>
inline const CalDet<T>& CalDet<T>::operator+= (const CalDet& other)
{
}

}
}

#endif
