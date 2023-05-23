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

#ifndef O2_CTP_RAWDECODER_H
#define O2_CTP_RAWDECODER_H

#include <vector>
#include <deque>
#include "DPLUtils/DPLRawParser.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/InputRecord.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "CTPReconstruction/RawDataDecoder.h"

namespace o2
{
namespace ctp
{
namespace reco_workflow
{

/// \class RawDecoderSpec
/// \brief Coverter task for Raw data to CTP digits
/// \author Roman Lietava from CPV example
///
class RawDecoderSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawDecoderSpec(bool digits, bool lumi) : mDoDigits(digits), mDoLumi(lumi) {}
  /// \brief Destructor
  ~RawDecoderSpec() override = default;
  /// \brief Initializing the RawDecoderSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  /// \brief Run conversion of raw data to cells
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input RawData: {"ROUT", "RAWDATA", 0, Lifetime::Timeframe}
  /// Output HW errors: {"CTP", "RAWHWERRORS", 0, Lifetime::Timeframe} -later
  void run(framework::ProcessingContext& ctx) final;

 protected:
 private:
  // for digits
  bool mDoDigits = true;
  std::vector<CTPDigit> mOutputDigits;
  // for lumi
  bool mDoLumi = true;
  //
  LumiInfo mOutputLumiInfo;
  bool mVerbose = false;
  uint64_t mCountsT = 0;
  uint64_t mCountsV = 0;
  uint32_t mNTFToIntegrate = 1;
  uint32_t mNHBIntegratedT = 0;
  uint32_t mNHBIntegratedV = 0;
  std::deque<size_t> mHistoryT;
  std::deque<size_t> mHistoryV;
  RawDataDecoder mDecoder;
};

/// \brief Creating DataProcessorSpec for the CTP
///
o2::framework::DataProcessorSpec getRawDecoderSpec(bool askSTFDist, bool digits, bool lumi);

} // namespace reco_workflow

} // namespace ctp

} // namespace o2

#endif
