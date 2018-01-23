// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SIM_TIMESTAMP_H
#define ALICEO2_SIM_TIMESTAMP_H

#include "Rtypes.h"

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
  TimeStamp() = default;
  TimeStamp(T time) { mTimeStamp = time; }
  T getTimeStamp() const { return mTimeStamp; }
  void setTimeStamp(T t) { mTimeStamp = t; }
 private:
  T mTimeStamp = 0;
  ClassDefNV(TimeStamp, 1);
};

template <typename T, typename E>
class TimeStampWithError : public TimeStamp<T>
{
 public:
  TimeStampWithError() = default;
  TimeStampWithError(T t, E te) : TimeStamp<T>(t), mTimeStampError(te) {}
  E getTimeStampError() const { return mTimeStampError; }
  void setTimeStampError(E te) { mTimeStampError = te; }
 private:
  E mTimeStampError = 0;
  ClassDefNV(TimeStampWithError, 1);
};
}
}

#endif /* ALICEO2_SIM_TIMESTAMP_H */
