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

namespace o2
{

namespace ctp
{

namespace reco_workflow
{

/// \class RawToDigitConverterSpec
/// \brief Coverter task for Raw data to CTP digits
/// \author Roman Lietava from CPV example
///
class RawToDigitConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawToDigitConverterSpec() = default;

  /// \brief Destructor
  ~RawToDigitConverterSpec() override = default;

  /// \brief Initializing the RawToDigitConverterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of raw data to cells
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input RawData: {"ROUT", "RAWDATA", 0, Lifetime::Timeframe}
  /// Output HW errors: {"CTP", "RAWHWERRORS", 0, Lifetime::Timeframe} -later
  void run(framework::ProcessingContext& ctx) final;
  static void makeGBTWordInverse(std::vector<gbtword80_t>& diglets, gbtword80_t& GBTWord, gbtword80_t& remnant, uint32_t& size_gbt, uint32_t Npld);

 protected:
 private:
  std::vector<CTPDigit> mOutputDigits;
};

/// \brief Creating DataProcessorSpec for the CTP
///
o2::framework::DataProcessorSpec getRawToDigitConverterSpec(bool askSTFDist);

} // namespace reco_workflow

} // namespace ctp

} // namespace o2
