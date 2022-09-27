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

/// \file   MID/Raw/src/Encoder.cxx
/// \brief  MID raw data encoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Encoder.h"

#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MIDRaw/GBTMapper.h"
#include "MIDRaw/Utils.h"
#include <fmt/format.h>

namespace o2
{
namespace mid
{

void Encoder::init(std::string_view outDir, std::string_view fileFor, int verbosity, std::vector<ROBoardConfig> configurations)
{
  /// Initializes links

  auto linkUniqueIds = mFEEIdConfig.getConfiguredLinkUniqueIDs();

  // Initialises the GBT link encoders
  for (auto& linkUniqueId : linkUniqueIds) {
    auto gbtUniqueId = mFEEIdConfig.getGBTUniqueId(linkUniqueId);
    std::vector<ROBoardConfig> gbtConfigs;
    for (auto& cfg : configurations) {
      if (gbtmapper::isBoardInGBT(cfg.boardId, gbtUniqueId)) {
        gbtConfigs.emplace_back(cfg);
      }
    }
    mGBTEncoders[gbtUniqueId].setConfig(gbtUniqueId, gbtConfigs);
  }

  // Initializes the output link
  auto ir = getOrbitIR(0);
  mRawWriter.setVerbosity(verbosity);
  for (uint16_t cruId = 0; cruId < 2; ++cruId) {
    for (uint8_t epId = 0; epId < 2; ++epId) {
      uint16_t feeId = 2 * cruId + epId;
      std::string outFileLink = fmt::format("{}/MID", outDir);
      if (fileFor != "all") { // single file for all links
        outFileLink += "_alio2-cr1-flp159";
        if (fileFor != "flp") {
          outFileLink += fmt::format("_cru{}_{}", cruId, epId);
          if (fileFor != "cruendpoint") {
            outFileLink += fmt::format("_lnk{}_feeid{}", raw::sUserLogicLinkID, feeId);
            if (fileFor != "link") {
              throw std::runtime_error("invalid option provided for file grouping");
            }
          }
        }
      }
      outFileLink += ".raw";
      mRawWriter.registerLink(feeId, cruId, raw::sUserLogicLinkID, epId, outFileLink);

      for (auto gbtUniqueId : mFEEIdConfig.getGBTUniqueIdsInLink(feeId)) {
        // Initializes the trigger response to be added to the empty HBs
        mGBTEncoders[gbtUniqueId].processTrigger(ir, raw::sORB);
        mGBTEncoders[gbtUniqueId].flush(mOrbitResponse[feeId], ir);
      }

      mOrbitResponseWord[feeId] = mOrbitResponse[feeId];
      completeWord(mOrbitResponseWord[feeId]);
    }
  }

  mRawWriter.setEmptyPageCallBack(this);
}

void Encoder::emptyHBFMethod(const o2::header::RDHAny* rdh, std::vector<char>& toAdd) const
{
  /// Response to orbit triggers in empty HBFs
  toAdd = mOrbitResponseWord[o2::raw::RDHUtils::getFEEID(rdh)];
}

void Encoder::onOrbitChange(uint32_t orbit)
{
  /// Performs action when orbit changes
  auto ir = getOrbitIR(orbit);
  for (uint16_t feeId = 0; feeId < 4; ++feeId) {
    // Write the data corresponding to the previous orbit
    writePayload(feeId, ir);
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

void Encoder::writePayload(uint16_t feeId, const InteractionRecord& ir, bool onlyNonEmpty)
{
  /// Writes data

  std::vector<char> buf = mOrbitResponse[feeId];
  for (auto& gbtUniqueId : mFEEIdConfig.getGBTUniqueIdsInLink(feeId)) {
    if (!mGBTEncoders[gbtUniqueId].isEmpty()) {
      mGBTEncoders[gbtUniqueId].flush(buf, ir);
    }
  }
  if (onlyNonEmpty && buf.size() == mOrbitResponse[feeId].size()) {
    return;
  }

  // Add the orbit response
  completeWord(buf);
  mRawWriter.addData(feeId, feeId / 2, raw::sUserLogicLinkID, feeId % 2, ir, buf);
}

void Encoder::finalize(bool closeFile)
{
  /// Writes remaining data and closes the file
  initIR();
  auto ir = getOrbitIR(mLastIR.orbit);
  auto nextIr = getOrbitIR(mLastIR.orbit + 1);
  for (uint16_t feeId = 0; feeId < 4; ++feeId) {
    // Write the last payload
    writePayload(feeId, ir, true);
    // Since the regional response comes after few clocks,
    // we might have the corresponding regional cards in the next orbit.
    // If this is the case, we flush all data of the next orbit
    writePayload(feeId, nextIr, true);
  }
  if (closeFile) {
    mRawWriter.close();
  }
}

void Encoder::process(gsl::span<const ColumnData> data, InteractionRecord ir, EventType eventType)
{
  /// Encodes data

  // The CTP trigger arrives to the electronics with a delay
  if (ir.differenceInBC(mRawWriter.getHBFUtils().getFirstSampledTFIR()) < mElectronicsDelay.localToBC) {
    // Due to the delay, these data would arrive in the TF before the first sampled one.
    // We therefore reject them.
    return;
  }
  applyElectronicsDelay(ir.orbit, ir.bc, -mElectronicsDelay.localToBC);

  initIR();

  if (ir.orbit != mLastIR.orbit) {
    onOrbitChange(mLastIR.orbit);
  }

  // Converts ColumnData to ROBoards
  mConverter.process(data);

  mGBTMap.clear();

  // Group local boards according to the GBT link they belong
  for (auto& item : mConverter.getDataMap()) {
    auto feeId = gbtmapper::getGBTIdFromUniqueLocId(item.first);
    mGBTMap[feeId].emplace_back(item.second);
  }

  // Process the GBT links
  for (auto& item : mGBTMap) {
    mGBTEncoders[item.first].process(item.second, ir);
  }
  mLastIR = ir;
}

void Encoder::initIR()
{
  if (mLastIR.isDummy()) {
    mLastIR = mRawWriter.getHBFUtils().getFirstSampledTFIR();
  }
}

} // namespace mid
} // namespace o2
