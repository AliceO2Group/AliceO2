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

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsCTP/Digits.h"
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{

namespace ctp
{

namespace lumi_workflow
{
struct lumiPoint
{
  lumiPoint() = default;
  InteractionRecord ir;   // timestamp of start of lumi interval
  float_t length = 1;        // length of interval in HB
  float_t counts = 0;         //  counts in the interval
  float_t getLumi() { return counts/length/88e-6; };
  float_t getFractErrorLumi() { return 1./sqrt(counts); };
};
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
  gbtword80_t mTVXMask = 0x4 ; // TVX is 3rd input
  std::vector<lumiPoint> mOutputLumiPoints;
};

/// \brief Creating DataProcessorSpec for the CTP
///
o2::framework::DataProcessorSpec getRawToLumiConverterSpec(bool askSTFDist);

} // namespace reco_workflow

} // namespace ctp

} // namespace o2
