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

#ifndef ALICEO2_TIMEFRAME_H
#define ALICEO2_TIMEFRAME_H

#include <FairMQParts.h>
#include "TObject.h" // for ClassDef
#include "Headers/TimeStamp.h"

namespace o2
{
namespace dataformats
{

using PartPosition = int;
typedef std::pair<o2::header::DataHeader, PartPosition> IndexElement;

// helper struct so that we can
// stream messages using ROOT
struct MessageSizePair {
  MessageSizePair() : size(0), buffer(nullptr) {}
  MessageSizePair(size_t s, char* b) : size(s), buffer(b) {}
  Int_t size;   // size of buffer in bytes (must be Int_t due to ROOT requirement)
  char* buffer; //[size]
};

// a class encapsulating a TimeFrame as sent out by EPN
class TimeFrame
{
 public:
  TimeFrame() : mParts() {} // in principle just for ROOT IO
  // constructor taking FairMQParts
  // might offer another constructor not depending on FairMQ ... directly taking buffers?
  // FIXME: take care of ownership later
  TimeFrame(FairMQParts& parts) : mParts()
  {
    for (int i = 0; i < parts.Size(); ++i) {
      mParts.emplace_back(parts[i].GetSize(), (char*)parts[i].GetData());
    }
  }

  // return TimeStamp (starttime) of this TimeFrame
  o2::header::TimeStamp const& GetTimeStamp() const { return mTimeStamp; }

  // return duration of this TimeFrame
  // allow user to ask for specific unit (needs to be std::chrono unit)
  template <typename TimeUnit>
  std::chrono::duration<typename TimeUnit::duration> GetDuration() const
  {
    // FIXME: implement
    return 0.;
  }

  // from how many flps we received data
  int GetNumFlps() const
  { /* FIXME: implement */
    return 0;
  }
  // is this TimeFrame complete
  bool IsComplete() const
  { /* FIXME: implement */
    return false;
  }
  // return the number of message parts in this TimeFrame
  size_t GetNumParts() const { return mParts.size(); }
  // access to the raw data
  MessageSizePair& GetPart(size_t i) { return mParts[i]; }
  // Get total payload size in bytes
  size_t GetPayloadSize() const
  { /* FIXME: implement */
    return 0;
  }
  // Get payload size of part i
  size_t GetPayloadSize(size_t i) const
  { /* FIXME: implement */
    return 0;
  }

 private:
  // FIXME: enable this when we have a dictionary for TimeStamp etc
  o2::header::TimeStamp mTimeStamp; //! the TimeStamp for this TimeFrame

  size_t mEpnId;                       // EPN origin of TimeFrame
  std::vector<MessageSizePair> mParts; // the message parts as accumulated by the EPN

  // FIXME: add Index data structure
  // Index mIndex; // index structure into parts

  ClassDefNV(TimeFrame, 1);
};
} // namespace dataformats
} // namespace o2

#endif // ALICEO2_TIMEFRAME_H
