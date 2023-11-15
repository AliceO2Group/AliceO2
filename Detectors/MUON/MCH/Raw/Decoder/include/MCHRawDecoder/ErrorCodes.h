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

#ifndef O2_MCH_RAW_ERROR_CODES_H
#define O2_MCH_RAW_ERROR_CODES_H

#include <string>
#include <cstdint>

namespace o2
{
namespace mch
{
namespace raw
{

enum ErrorCodes {
  ErrorParity = 1,                           // 1
  ErrorHammingCorrectable = 1 << 1,          // 2
  ErrorHammingUncorrectable = 1 << 2,        // 4
  ErrorBadSyncPacket = 1 << 3,               // 8
  ErrorBadHeartBeatPacket = 1 << 4,          // 16
  ErrorBadDataPacket = 1 << 5,               // 32
  ErrorBadClusterSize = 1 << 6,              // 64
  ErrorBadIncompleteWord = 1 << 7,           // 128
  ErrorTruncatedData = 1 << 8,               // 256
  ErrorTruncatedDataUL = 1 << 9,             // 512
  ErrorUnexpectedSyncPacket = 1 << 10,       // 1024
  ErrorBadELinkID = 1 << 11,                 // 2048
  ErrorBadLinkID = 1 << 12,                  // 4096
  ErrorUnknownLinkID = 1 << 13,              // 8192
  ErrorBadHBTime = 1 << 14,                  // 16384
  ErrorNonRecoverableDecodingError = 1 << 15 // 32768
};

uint32_t getErrorCodesSize();

std::string errorCodeAsString(uint32_t code);

} // namespace raw
} // namespace mch
} // namespace o2

#endif
