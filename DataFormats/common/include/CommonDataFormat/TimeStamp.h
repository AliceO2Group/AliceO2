// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_COMMON_TIMESTAMP_H
#define ALICEO2_COMMON_TIMESTAMP_H

#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace dataformats
{
// A minimal TimeStamp class for simulation in the spirit of FairTimeStamp
// but without having FairLinks attached
template <typename T>
class TimeStamp
{
 public:
  GPUdDefault() TimeStamp() CON_DEFAULT;
  GPUdDefault() ~TimeStamp() CON_DEFAULT;
  GPUdi() TimeStamp(T time) { mTimeStamp = time; }
  GPUdi() T getTimeStamp() const { return mTimeStamp; }
  GPUdi() void setTimeStamp(T t) { mTimeStamp = t; }
  GPUdi() bool operator==(const TimeStamp<T>& t) const { return mTimeStamp == t.mTimeStamp; }

 private:
  T mTimeStamp = 0;
  ClassDefNV(TimeStamp, 1);
};

template <typename T, typename E>
class TimeStampWithError : public TimeStamp<T>
{
 public:
  GPUdDefault() TimeStampWithError() = default;
  GPUd() TimeStampWithError(T t, E te) : TimeStamp<T>(t), mTimeStampError(te) {}
  GPUdi() E getTimeStampError() const { return mTimeStampError; }
  GPUdi() void setTimeStampError(E te) { mTimeStampError = te; }

 private:
  E mTimeStampError = 0;
  ClassDefNV(TimeStampWithError, 1);
};

#ifndef GPUCA_GPUCODE
template <typename T>
std::ostream& operator<<(std::ostream& os, const TimeStamp<T>& t)
{
  // stream itself
  os << t.getTimeStamp();
  return os;
}

template <typename T, typename E>
std::ostream& operator<<(std::ostream& os, const TimeStampWithError<T, E>& t)
{
  // stream itself
  os << t.getTimeStamp() << " +/- " << t.getTimeStampError();
  return os;
}
#endif
} // namespace dataformats
} // namespace o2

#endif /* ALICEO2_COMMON_TIMESTAMP_H */
