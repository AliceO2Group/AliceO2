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
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "PHOSBase/Mapping.h"
#include "PHOSCalib/CalibParams.h"
#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSReconstruction/RawReaderError.h"

namespace o2
{

namespace phos
{

namespace reco_workflow
{

/// \class RawToCellConverterSpec
/// \brief Coverter task for Raw data to PHOS cells
/// \author Dmitri Peresunko NRC KI
/// \since Sept., 2020
///
class RawToCellConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawToCellConverterSpec() : framework::Task(){};

  /// \brief Destructor
  ~RawToCellConverterSpec() override = default;

  /// \brief Initializing the RawToCellConverterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of raw data to cells
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input RawData: {"ROUT", "RAWDATA", 0, Lifetime::Timeframe}
  /// Output cells: {"PHS", "CELLS", 0, Lifetime::Timeframe}
  /// Output cells trigger record: {"PHS", "CELLSTR", 0, Lifetime::Timeframe}
  /// Output HW errors: {"PHS", "RAWHWERRORS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 protected:
  /// \brief simple check of HW address
  char CheckHWAddress(short ddl, short hwAddress, short& fee);

 private:
  bool mFillChi2 = false;                                     ///< Fill output with quality of samples
  bool mCombineGHLG = true;                                   ///< Combine or not HG and LG channels (def: combine, LED runs: not combine)
  bool mPedestalRun = false;                                  ///< Analyze pedestal run (calculate pedestal mean and RMS)
  std::unique_ptr<Mapping> mMapping;                          ///< Mapping
  std::unique_ptr<CalibParams> mCalibParams;                  ///!<! PHOS calibration
  std::unique_ptr<CaloRawFitter> mRawFitter;                  ///!<! Raw fitter
  std::vector<o2::phos::Cell> mOutputCells;                   ///< Container with output cells
  std::vector<o2::phos::TriggerRecord> mOutputTriggerRecords; ///< Container with output cells
  std::vector<o2::phos::RawReaderError> mOutputHWErrors;      ///< Errors occured in reading data
  std::vector<short> mOutputFitChi;                           ///< Raw sample fit quality
};

/// \brief Creating DataProcessorSpec for the PHOS Cell Converter Spec
///
/// Refer to RawToCellConverterSpec::run for input and output specs
framework::DataProcessorSpec getRawToCellConverterSpec();

} // namespace reco_workflow

} // namespace phos

} // namespace o2
