// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/Encoder.cxx
/// \brief  MID raw data encoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Encoder.h"

#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MIDRaw/CrateMasks.h"
#include <fmt/format.h>

namespace o2
{
namespace mid
{

void Encoder::init(const char* filename, bool perLink, int verbosity, bool debugMode)
{
  /// Initializes links

  CrateMasks masks;
  auto gbtIds = mFEEIdConfig.getConfiguredGBTIds();

  mRawWriter.setVerbosity(verbosity);
  int lcnt = 0;
  for (auto& gbtId : gbtIds) {
    auto feeId = mFEEIdConfig.getFeeId(gbtId);
    mRawWriter.registerLink(feeId, mFEEIdConfig.getCRUId(gbtId), raw::sUserLogicLinkID, mFEEIdConfig.getEndPointId(gbtId), perLink ? fmt::format("{:s}_L{:d}.raw", filename, lcnt) : fmt::format("{:s}.raw", filename));
    mGBTEncoders[feeId].setFeeId(feeId);
    mGBTEncoders[feeId].setMask(masks.getMask(feeId));
    mGBTIds[feeId] = gbtId;
    lcnt++;

    // Initializes the trigger response to be added to the empty HBs
    auto ir = getOrbitIR(0);
    mGBTEncoders[feeId].processTrigger(ir, raw::sORB);
    mGBTEncoders[feeId].flush(mOrbitResponse[feeId], ir);
    mOrbitResponseWord[feeId] = mOrbitResponse[feeId];
    completeWord(mOrbitResponseWord[feeId]);
  }

  mRawWriter.setEmptyPageCallBack(this);

  mConverter.setDebugMode(debugMode);
}

void Encoder::emptyHBFMethod(const o2::header::RDHAny* rdh, std::vector<char>& toAdd) const
{
  /// Response to orbit triggers in empty HBFs
  auto feeId = o2::raw::RDHUtils::getFEEID(rdh);
  toAdd = mOrbitResponseWord[feeId];
}

void Encoder::onOrbitChange(uint32_t orbit)
{
  /// Performs action when orbit changes
  auto ir = getOrbitIR(orbit);
  for (uint16_t feeId = 0; feeId < crateparams::sNGBTs; ++feeId) {
    // Write the data corresponding to the previous orbit
    if (!mGBTEncoders[feeId].isEmpty()) {
      writePayload(feeId, ir);
    }
  }
}

void Encoder::completeWord(std::vector<char>& buffer)
{
  /// Completes the buffer with zeros to reach the expected CRU word size
  size_t dataSize = buffer.size();
  size_t cruWord = 2 * o2::raw::RDHUtils::GBTWord;
  size_t modulo = dataSize % cruWord;
  if (modulo) {
    dataSize += cruWord - modulo;
    buffer.resize(dataSize, static_cast<char>(0));
  }
}

void Encoder::writePayload(uint16_t feeId, const InteractionRecord& ir)
{
  /// Writes data

  // The first part of data is the answer to the orbit trigger
  std::vector<char> buf = mOrbitResponse[feeId];
  // Then we flush the received data
  mGBTEncoders[feeId].flush(buf, ir);
  completeWord(buf);
  mRawWriter.addData(feeId, mFEEIdConfig.getCRUId(mGBTIds[feeId]), raw::sUserLogicLinkID, mFEEIdConfig.getEndPointId(mGBTIds[feeId]), ir, buf);
}

void Encoder::finalize(bool closeFile)
{
  /// Writes remaining data and closes the file
  if (mLastIR.isDummy()) {
    mLastIR.bc = 0;
    mLastIR.orbit = mRawWriter.getHBFUtils().orbitFirst;
  }
  auto ir = getOrbitIR(mLastIR.orbit);
  for (uint16_t feeId = 0; feeId < crateparams::sNGBTs; ++feeId) {
    // Write the last payload
    writePayload(feeId, ir);
    if (!mGBTEncoders[feeId].isEmpty()) {
      // Since the regional response comes after few clocks,
      // we might have the corresponding regional cards in the next orbit.
      // If this is the case, we flush all data of the next orbit
      ++ir.orbit;
      writePayload(feeId, ir);
    }
  }
  if (closeFile) {
    mRawWriter.close();
  }
}

void Encoder::process(gsl::span<const ColumnData> data, const InteractionRecord& ir, EventType eventType)
{
  /// Encodes data
  if (ir.orbit != mLastIR.orbit) {
    onOrbitChange(mLastIR.orbit);
  }

  mConverter.process(data);

  for (auto& item : mConverter.getData()) {
    mGBTEncoders[item.first].process(item.second, ir);
  }
  mLastIR = ir;
}
} // namespace mid
} // namespace o2
