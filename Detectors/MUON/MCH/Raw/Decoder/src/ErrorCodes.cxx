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

#include "MCHRawDecoder/ErrorCodes.h"

namespace o2
{
namespace mch
{
namespace raw
{

uint32_t getErrorCodesSize()
{
  return 15;
}

void append(const char* msg, std::string& to)
{
  std::string s = to.size() ? "& " + to : "";
  s += msg;
  if (to.size()) {
    to += "& ";
  }
  to += msg;
}

std::string errorCodeAsString(uint32_t ec)
{
  std::string msg;

  if (ec & ErrorParity) {
    append("Parity", msg);
  }
  if (ec & ErrorHammingCorrectable) {
    append("Hamming Correctable", msg);
  }
  if (ec & ErrorHammingUncorrectable) {
    append("Hamming Uncorrectable", msg);
  }
  if (ec & ErrorBadClusterSize) {
    append("Cluster Size", msg);
  }
  if (ec & ErrorBadSyncPacket) {
    append("Bad Sync Packet", msg);
  }
  if (ec & ErrorUnexpectedSyncPacket) {
    append("Unexpected Sync", msg);
  }
  if (ec & ErrorBadHeartBeatPacket) {
    append("Bad HB Packet", msg);
  }
  if (ec & ErrorBadDataPacket) {
    append("Bad Data Packet", msg);
  }
  if (ec & ErrorBadIncompleteWord) {
    append("Bad Incomplete Word", msg);
  }
  if (ec & ErrorTruncatedData) {
    append("Truncated Data", msg);
  }
  if (ec & ErrorBadELinkID) {
    append("Bad E-Link ID", msg);
  }
  if (ec & ErrorBadLinkID) {
    append("Bad Link ID", msg);
  }
  if (ec & ErrorUnknownLinkID) {
    append("Unknown Link ID", msg);
  }
  if (ec & ErrorBadHBTime) {
    append("Bad HB Time", msg);
  }
  if (ec & ErrorNonRecoverableDecodingError) {
    append("Non Recoverable", msg);
  }
  return msg;
}

} // namespace raw
} // namespace mch
} // namespace o2
