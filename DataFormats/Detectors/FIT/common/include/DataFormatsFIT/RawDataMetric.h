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
//
// file RawDataMetric.h class  Struct for collecting metrics during decoding
// with Artur.Furs@cern.ch
//
#ifndef ALICEO2_FIT_RAWDATAMETRIC_H_
#define ALICEO2_FIT_RAWDATAMETRIC_H_

#include <array>
#include <map>
#include <string>
#include <cstdint>

namespace o2
{
namespace fit
{
struct RawDataMetric {
  RawDataMetric(uint8_t linkID, uint8_t EPID, uint16_t FEEID, bool isRegisteredFEE = true) : mLinkID(linkID), mEPID(EPID), mFEEID(FEEID), mIsRegisteredFEE(isRegisteredFEE) {}
  ~RawDataMetric() = default;

  enum EStatusBits {
    kIncompletePayload,   // Incomplete payload
    kWrongDescriptor,     // Wrong descriptors in header
    kWrongChannelMapping, // Wrong channel mapping
    kEmptyDataBlock,      // Only header in data block
    kDecodedDataBlock     // Decoded w/o any issue data block
  };
  typedef uint8_t Status_t;
  constexpr static uint8_t sNbits = 5;
  inline bool checkBadDataBlock(Status_t metric)
  {
    bool result = checkStatusBit(metric, EStatusBits::kIncompletePayload);
    if (!result) { // Incomplete payload has high priority among errors
      result |= checkStatusBit(metric, EStatusBits::kEmptyDataBlock);
      // Lets just check this for a while, w/o any decision for data block
      // result |= checkStatusBit(metric,EStatusBits::kWrongDescriptor);
      checkStatusBit(metric, EStatusBits::kWrongDescriptor);
    }
    return result;
  }
  inline void addStatusBit(EStatusBits statusBit, bool val = true)
  {
    mBitStats[statusBit] += static_cast<int>(val);
  }

  inline bool checkStatusBit(Status_t metric, EStatusBits statusBit)
  {
    const bool result = (metric & (1 << statusBit)) > 0;
    mBitStats[statusBit] += static_cast<int>(result);
    return result;
  }

  inline static bool isBitActive(Status_t metric, EStatusBits statusBit)
  {
    return (metric & (1 << statusBit)) > 0;
  }

  inline static void setStatusBit(Status_t& metric, EStatusBits statusBit, bool val = true)
  {
    metric |= (static_cast<uint8_t>(val) << statusBit);
  }
  void print() const;
  static Status_t getAllBitsActivated();
  uint8_t mLinkID;
  uint8_t mEPID;
  uint16_t mFEEID;
  bool mIsRegisteredFEE;
  std::array<std::size_t, sNbits> mBitStats{};
  const static std::map<unsigned int, std::string> sMapBitsToNames;
};
} // namespace fit
} // namespace o2
#endif
