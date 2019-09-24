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

#include "DataFormatsEMCAL/Cell.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \class CellConverterSpec
/// \brief Coverter task for EMCAL digits to EMCAL cells
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Sept 24, 2019
///
/// Task converting a vector of EMCAL digits to a vector of
/// EMCAL cells. No selection of digits is done, digits are
/// copied 1-to-1 to the output. If MC is available and requested,
/// the MC truth container is ported to the output. Refer to
/// run for the list of input and output specs.
class CellConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  CellConverterSpec(bool propagateMC) : framework::Task(), mPropagateMC(propagateMC){};

  /// \brief Destructor
  ~CellConverterSpec() override = default;

  /// \brief Initializing the CellConverterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of digits to cells
  /// \param ctx Processing context
  ///
  /// Converting the input vector of o2::emcal::Digit to
  /// an output vector of o2::emcal::Cell. Data is copied
  /// 1-to-1, no additional selection is done. In case
  /// the task is configured to propagate MC-truth the
  /// MC truth container is propagated 1-to-1 to the output.
  ///
  /// The following branches are linked:
  /// Input digits: {"EMC", "DIGITS", 0, Lifetime::Timeframe}
  /// Input MC-truth: {"EMC", "DIGITSMCTR", 0, Lifetime::Timeframe}
  /// Output cells: {"EMC", "CELLS", 0, Lifetime::Timeframe}
  /// Output MC-truth: {"EMC", "CELLSMCTR", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  bool mPropagateMC = false;                 ///< Switch whether to process MC true labels
  std::vector<o2::emcal::Cell> mOutputCells; ///< Container with output cells
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
/// \param propagateMC If true the MC truth container is propagated to the output
///
/// Refer to CellConverterSpec::run for input and output specs
framework::DataProcessorSpec getCellConverterSpec(bool propagateMC);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
