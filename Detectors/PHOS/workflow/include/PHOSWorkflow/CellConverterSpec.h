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

#include "DataFormatsPHOS/BadChannelsMap.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace phos
{

namespace reco_workflow
{

/// \class CellConverterSpec
/// \brief Coverter task for PHOS digits to AOD PHOS cells
/// \author Dmitri Peresunko after Markus Fasel
/// \since Dec 14, 2019
///
/// Task converting a vector of PHOS digits to a vector of
/// PHOS cells. No selection of digits is done, digits are
/// copied 1-to-1 to the output. If MC is available and requested,
/// the MC truth container is ported to the output. Refer to
/// run for the list of input and output specs.
class CellConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  CellConverterSpec(bool propagateMC, bool defBadMap) : framework::Task(), mPropagateMC(propagateMC), mDefBadMap(defBadMap){};

  /// \brief Destructor
  ~CellConverterSpec() override = default;

  /// \brief Initializing the CellConverterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of digits to cells
  /// \param ctx Processing context
  ///
  /// Converting the input vector of o2::phos::Digit to
  /// an output vector of o2::phos::Cell. Data is copied
  /// 1-to-1, no additional selection is done. In case
  /// the task is configured to propagate MC-truth the
  /// MC truth container is propagated 1-to-1 to the output.
  ///
  /// The following branches are linked:
  /// Input digits: {"PHS", "DIGITS", 0, Lifetime::Timeframe}
  /// Input digit trigger records: {"PHS", "DIGITTRS", 0, Lifetime::Timeframe}
  /// Input MC-truth: {"PHS", "DIGITSMCTR", 0, Lifetime::Timeframe}
  /// Output cells: {"PHS", "CELLS", 0, Lifetime::Timeframe}
  /// Output cell trigger records: {"PHS", "CELLTRS", 0, Lifetime::Timeframe}
  /// Output MC-truth: {"PHS", "CELLSMCTR", 0, Lifetime::Timeframe}
  /// Output MC-map:   {"PHS", "CELLSMCMAP", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  bool mPropagateMC = false;                                   ///< Switch whether to process MC true labels
  bool mDefBadMap = false;                                     ///< Use default bad map and calibration or extract from CCDB
  bool mHasCalib = false;                                      ///< Were calibration objects received
  bool mInitSimParams = true;                                  ///< Sim/RecoParams to be initialized
  std::vector<Cell> mOutputCells;                              ///< Container with output cells
  std::vector<TriggerRecord> mOutputCellTrigRecs;              ///< Container with trigger records for output cells
  o2::dataformats::MCTruthContainer<MCLabel> mOutputTruthCont; ///< output MC labels
  std::unique_ptr<BadChannelsMap> mBadMap;                     ///< Bad map
};

/// \brief Creating DataProcessorSpec for the PHOS Cell Converter Spec
/// \param propagateMC If true the MC truth container is propagated to the output
///
/// Refer to CellConverterSpec::run for input and output specs
framework::DataProcessorSpec getCellConverterSpec(bool propagateMC, bool defBadMap);

} // namespace reco_workflow

} // namespace phos

} // namespace o2
