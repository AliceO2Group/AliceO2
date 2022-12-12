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

#ifndef ALICEO2_FOCAL_PADWORD_H
#define ALICEO2_FOCAL_PADWORD_H

#include <iosfwd>
#include <gsl/span>
#include <cstdint>

namespace o2::focal
{

struct PadASICWord {
  union {
    struct {
      uint32_t mTrailer : 3;
      uint32_t mWADD : 9;
      uint32_t mBCID : 12;
      uint32_t mFourbit : 4;
      uint32_t mHeader : 4;
    }; // ASIC Header word
    struct {
      uint32_t mADC : 10;
      uint32_t mTOA : 10;
      uint32_t mTOT : 12;
    }; // ASIC channel word
    uint32_t mData = 0;
  };
};

struct TriggerWord {
  union {
    struct {
      uint64_t mHeader : 8;
      uint64_t mTrigger0 : 7;
      uint64_t mTrigger1 : 7;
      uint64_t mTrigger2 : 7;
      uint64_t mTrigger3 : 7;
      uint64_t mTrigger4 : 7;
      uint64_t mTrigger5 : 7;
      uint64_t mTrigger6 : 7;
      uint64_t mTrigger7 : 7;
    };
    uint64_t mData = 0;
  };
};

struct ASICHeader : public PadASICWord {
  ASICHeader()
  {
    mData = 0;
  }
  ASICHeader(uint32_t word)
  {
    mData = word;
  }
  ASICHeader(uint32_t header, uint32_t wadd, uint32_t, uint32_t bcID, uint32_t fourbit, uint32_t trailer)
  {
    mTrailer = trailer;
    mWADD = wadd;
    mBCID = bcID;
    mFourbit = fourbit;
    mHeader = header;
  }
  uint32_t getTrailer() const { return mTrailer; }
  uint32_t getWadd() const { return mWADD; }
  uint32_t getBCID() const { return mBCID; }
  uint32_t getFourbit() const { return mFourbit; }
  uint32_t getHeader() const { return mHeader; }
};

struct ASICChannel : public PadASICWord {
  ASICChannel()
  {
    mData = 0;
  }
  ASICChannel(uint32_t word)
  {
    mData = 0;
  }
  ASICChannel(uint32_t adc, uint32_t toa, uint32_t tot)
  {
    mADC = adc;
    mTOA = toa;
    mTOT = tot;
  }
  uint32_t getADC() const { return mADC; }
  uint32_t getTOA() const { return mTOA; }
  uint32_t getTOT() const { return mTOT; }
};

struct PadGBTWord {
  union {
    uint64_t mTriggerWords[2];
    uint32_t mASICWords[4];
  };

  const TriggerWord& getTriggerData() const { return reinterpret_cast<const TriggerWord&>(mTriggerWords[0]); }

  template <typename T>
  gsl::span<const T> getASICData() const
  {
    return gsl::span<const T>(reinterpret_cast<const T*>(mASICWords), 2);
  }
};

std::ostream& operator<<(std::ostream& stream, const ASICChannel& channel);
std::ostream& operator<<(std::ostream& stream, const ASICHeader& header);
std::ostream& operator<<(std::ostream& stream, const TriggerWord& trigger);

} // namespace o2::focal

#endif // ALICEO2_FOCAL_PADWORD_H