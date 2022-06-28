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
#include <chrono>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ConcreteDataMatcher.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DataFormatsCPV/CalibParams.h"
#include "DataFormatsCPV/BadChannelMap.h"
#include "DataFormatsCPV/Pedestals.h"
#include "CPVReconstruction/RawDecoder.h"
#include "CCDB/BasicCCDBManager.h"

namespace o2
{

namespace cpv
{

namespace reco_workflow
{

/// \class RawToDigitConverterSpec
/// \brief Coverter task for Raw data to CPV cells
/// \author Dmitri Peresunko NRC KI
/// \since Sept., 2020
///
class RawToDigitConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawToDigitConverterSpec(bool isPedestal, bool useBadChannelMap, bool useGainCalibration) : framework::Task(),
                                                                                             mIsUsingGainCalibration(useGainCalibration),
                                                                                             mIsUsingBadMap(useBadChannelMap),
                                                                                             mIsPedestalData(isPedestal){};

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
  /// Output cells: {"CPV", "DIGITS", 0, Lifetime::Timeframe}
  /// Output cells trigger record: {"CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe}
  /// Output HW errors: {"CPV", "RAWHWERRORS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 protected:
  /// \brief simple check of HW address
  char CheckHWAddress(short ddl, short hwAddress, short& fee);

 private:
  bool mIsUsingGainCalibration;                                      ///< Use gain calibration from CCDB
  bool mIsUsingBadMap;                                               ///< Use BadChannelMap to mask bad channels
  bool mIsPedestalData;                                              ///< Do not subtract pedestals if true
  std::vector<Digit> mOutputDigits;                                  ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords;                  ///< Container with output cells
  std::vector<RawDecoderError> mOutputHWErrors;                      ///< Errors occured in reading data
  bool mIsMuteDecoderErrors = false;                                 ///< mute errors for 10 minutes
  int mDecoderErrorsCounterWhenMuted = 0;                            ///< errors counter while errors are muted
  int mDecoderErrorsPerMinute = 0;                                   ///< errors per minute counter
  int mMinutesPassed = 0;                                            ///< runtime duration in minutes
  std::chrono::time_point<std::chrono::system_clock> mStartTime;     ///< Time of start of decoding
  std::chrono::time_point<std::chrono::system_clock> mTimeWhenMuted; ///< Time when muted errors
};

/// \brief Creating DataProcessorSpec for the CPV Digit Converter Spec
///
/// Refer to RawToDigitConverterSpec::run for input and output specs
o2::framework::DataProcessorSpec getRawToDigitConverterSpec(bool askDISTSTF = true,
                                                            bool isPedestal = false,
                                                            bool useBadChannelMap = true,
                                                            bool useGainCalibration = true);

} // namespace reco_workflow

} // namespace cpv

} // namespace o2
