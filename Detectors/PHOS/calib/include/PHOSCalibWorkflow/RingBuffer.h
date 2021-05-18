// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_PHOS_RINGBUFFER_H
#define O2_CALIBRATION_PHOS_RINGBUFFER_H

/// @file   RingBuffer.h
/// @brief  Device to collect energy and time PHOS energy and time calibration.

#include <TLorentzVector.h>
#include <TVector3.h>
#include <array>

namespace o2
{
namespace phos
{

// For real/mixed distribution calculation
class RingBuffer
{
 public:
  RingBuffer() = default;
  ~RingBuffer() = default;

  short size()
  {
    if (mFilled) {
      return kBufferSize;
    } else {
      return mCurrent;
    }
  }
  void addEntry(TLorentzVector& v)
  {
    mBuffer[mCurrent] = v;
    mCurrent++;
    if (mCurrent >= kBufferSize) {
      mFilled = true;
      mCurrent -= kBufferSize;
    }
  }
  const TLorentzVector& getEntry(short index) const
  {
    //get entry from (mCurrent-1) corresponding to index=size()-1 down to size
    if (mFilled) {
      index += mCurrent;
    }
    index = index % kBufferSize;
    return mBuffer[index];
  }
  //mark that next added entry will be from next event
  void startNewEvent() { mStartCurrentEvent = mCurrent; }

  bool isCurrentEvent(short index) const
  {
    if (mCurrent >= mStartCurrentEvent) {
      return (index >= mStartCurrentEvent && index < mCurrent);
    } else {
      return (index >= mStartCurrentEvent || index < mCurrent);
    }
  }

 private:
  static constexpr short kBufferSize = 100;        ///< Total size of the buffer
  std::array<TLorentzVector, kBufferSize> mBuffer; ///< buffer
  bool mFilled = false;                            ///< if buffer fully filled
  short mCurrent = 0;                              ///< where next object will be added
  short mStartCurrentEvent = 0;                    ///< start of current event
};
} // namespace phos
} // namespace o2

#endif
