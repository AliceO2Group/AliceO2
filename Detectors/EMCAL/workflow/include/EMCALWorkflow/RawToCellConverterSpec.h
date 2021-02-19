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
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALReconstruction/CaloRawFitter.h"

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \class RawToCellConverterSpec
/// \brief Coverter task for Raw data to EMCAL cells
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since December 10, 2019
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
  /// Output cells: {"EMC", "CELLS", 0, Lifetime::Timeframe}
  /// Output cells trigger record: {"EMC", "CELLSTR", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

  void setNoiseThreshold(int thresold) { mNoiseThreshold = thresold; }
  int getNoiseThreshold() { return mNoiseThreshold; }

 private:
  int mNoiseThreshold = 0;
  o2::emcal::Geometry* mGeometry = nullptr;                     ///!<! Geometry pointer
  std::unique_ptr<o2::emcal::MappingHandler> mMapper = nullptr; ///!<! Mapper
  std::unique_ptr<o2::emcal::CaloRawFitter> mRawFitter;         ///!<! Raw fitter
  std::vector<o2::emcal::Cell> mOutputCells;                    ///< Container with output cells
  std::vector<o2::emcal::TriggerRecord> mOutputTriggerRecords;  ///< Container with output cells
  std::vector<int> mOutputDecoderErrors;                        ///< Container with decoder errors
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
///
/// Refer to RawToCellConverterSpec::run for input and output specs
framework::DataProcessorSpec getRawToCellConverterSpec();

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
