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

#include <vector>
#include <deque>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{

namespace ctp
{

namespace lumi_workflow
{

/// \class RawToLumiConverterSpec
/// \brief Coverter task for Raw data to Lumi
/// \author Roman Lietava from RawToDigiConverterSpec example
///
class RawToLumiConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawToLumiConverterSpec() = default;

  /// \brief Destructor
  ~RawToLumiConverterSpec() override = default;

  /// \brief Initializing the RawToDigitConverterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of raw data to cells
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input RawData: {"ROUT", "RAWDATA", 0, Lifetime::Timeframe}
  /// Output HW errors: {"CTP", "LUMI", 0, Lifetime::Timeframe} -later
  void run(framework::ProcessingContext& ctx) final;

 protected:
 private:
  gbtword80_t mTVXMask = 0x4; // TVX is 3rd input
  LumiInfo mOutputLumiInfo;
  size_t mCounts = 0;
  size_t mNTFToIntegrate = 1;
  size_t mNHBIntegrated = 0;
  std::deque<size_t> mHistory;
};

/// \brief Creating DataProcessorSpec for the CTP
///
o2::framework::DataProcessorSpec getRawToLumiConverterSpec(bool askSTFDist);

} // namespace lumi_workflow

} // namespace ctp

} // namespace o2
