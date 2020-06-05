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

namespace o2
{
namespace mid
{

void Encoder::init(const char* filename, int verbosity)
{
  /// Initializes links

  CrateMasks masks;
  auto gbtIds = mFEEIdConfig.getConfiguredGBTIds();

  mRawWriter.setVerbosity(verbosity);
  for (auto& gbtId : gbtIds) {
    auto feeId = mFEEIdConfig.getFeeId(gbtId);
    mRawWriter.registerLink(feeId, mFEEIdConfig.getCRUId(gbtId), mFEEIdConfig.getLinkId(gbtId), mFEEIdConfig.getEndPointId(gbtId), filename);
    mGBTEncoders[feeId].setFeeId(feeId);
    mGBTEncoders[feeId].setMask(masks.getMask(feeId));
    mGBTIds[feeId] = gbtId;
  }
}

void Encoder::hbTrigger(const InteractionRecord& ir)
{
  /// Processes HB trigger
  if (mLastIR.isDummy()) {
    // This is the first HB
    for (uint16_t feeId = 0; feeId < crateparams::sNGBTs; ++feeId) {
      mGBTEncoders[feeId].processTrigger(o2::constants::lhc::LHCMaxBunches, raw::sORB);
    }
    mLastIR = o2::raw::HBFUtils::Instance().getFirstIR();
    return;
  }

  std::vector<InteractionRecord> HBIRVec;
  o2::raw::HBFUtils::Instance().fillHBIRvector(HBIRVec, mLastIR + 1, ir);
  for (auto& hbIr : HBIRVec) {
    for (uint16_t feeId = 0; feeId < crateparams::sNGBTs; ++feeId) {
      flush(feeId, mLastIR);
      mGBTEncoders[feeId].processTrigger(o2::constants::lhc::LHCMaxBunches, raw::sORB);
    }
    mLastIR = hbIr;
  }
  mLastIR = ir;
}

void Encoder::flush(uint16_t feeId, const InteractionRecord& ir)
{
  /// Flushes data

  if (mGBTEncoders[feeId].getBufferSize() == 0) {
    return;
  }
  size_t dataSize = mGBTEncoders[feeId].getBufferSize();
  size_t resto = dataSize % o2::raw::RDHUtils::GBTWord;
  if (dataSize % o2::raw::RDHUtils::GBTWord) {
    dataSize += o2::raw::RDHUtils::GBTWord - resto;
  }
  std::vector<char> buf(dataSize);
  memcpy(buf.data(), mGBTEncoders[feeId].getBuffer().data(), mGBTEncoders[feeId].getBufferSize());
  mRawWriter.addData(feeId, mFEEIdConfig.getCRUId(mGBTIds[feeId]), mFEEIdConfig.getLinkId(mGBTIds[feeId]), mFEEIdConfig.getEndPointId(mGBTIds[feeId]), ir, buf);
  mGBTEncoders[feeId].clear();
}

void Encoder::finalize(bool closeFile)
{
  /// Finish the flushing and closes the
  for (uint16_t feeId = 0; feeId < crateparams::sNGBTs; ++feeId) {
    flush(feeId, mLastIR);
  }
  if (closeFile) {
    mRawWriter.close();
  }
}

void Encoder::process(gsl::span<const ColumnData> data, const InteractionRecord& ir, EventType eventType)
{
  /// Encodes data
  hbTrigger(ir);

  mConverter.process(data);

  for (auto& item : mConverter.getData()) {
    mGBTEncoders[item.first].process(item.second, ir.bc);
  }
}
} // namespace mid
} // namespace o2
