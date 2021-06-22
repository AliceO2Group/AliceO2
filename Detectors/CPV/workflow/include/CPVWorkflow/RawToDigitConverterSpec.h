// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
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
  RawToDigitConverterSpec() : framework::Task(){};

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
  bool mIsPedestalData;                             ///< Do not subtract pedestals if true
  bool mIsUsingCcdbMgr;                             ///< Are we using CCDB manager?
  long mCurrentTimeStamp;                           ///< Current timestamp for CCDB querying
  std::unique_ptr<CalibParams> mCalibParams;        ///< CPV calibration
  std::unique_ptr<Pedestals> mPedestals;            ///< CPV pedestals
  std::unique_ptr<BadChannelMap> mBadMap;           ///< BadMap
  std::vector<Digit> mOutputDigits;                 ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords; ///< Container with output cells
  std::vector<RawDecoderError> mOutputHWErrors;     ///< Errors occured in reading data
};

/// \brief Creating DataProcessorSpec for the CPV Digit Converter Spec
///
/// Refer to RawToDigitConverterSpec::run for input and output specs
o2::framework::DataProcessorSpec getRawToDigitConverterSpec(bool askDISTSTF = true);

} // namespace reco_workflow

} // namespace cpv

} // namespace o2
