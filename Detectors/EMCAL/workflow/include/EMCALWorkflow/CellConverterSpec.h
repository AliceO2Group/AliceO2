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

#ifndef O2_EMCAL_CELLCONVERTER_SPEC
#define O2_EMCAL_CELLCONVERTER_SPEC

#include <vector>

#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/MCLabel.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "EMCALBase/Geometry.h"
#include "EMCALReconstruction/CaloRawFitter.h"
#include "EMCALReconstruction/AltroHelper.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \class CellConverterSpec
/// \brief Coverter task for EMCAL digits to EMCAL cells
/// \ingroup EMCALworkflow
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
  /// \param useccdb If true the TecoParams
  /// \param inputSubsepc Subsepc of input objects
  /// \param outputSubspec Subspec of output objects
  CellConverterSpec(bool propagateMC, bool useccdb, int inputSubsepc, int outputSubspec) : framework::Task(), mPropagateMC(propagateMC), mLoadRecoParamFromCCDB(useccdb), mSubspecificationIn(inputSubsepc), mSubspecificationOut(outputSubspec){};

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
  /// Input trigers: {"EMC", "DIGITSTRGR", 0, Lifetime::Timeframe}
  /// Input MC-truth: {"EMC", "DIGITSMCTR", 0, Lifetime::Timeframe}
  /// Output cells: {"EMC", "CELLS", 0, Lifetime::Timeframe}
  /// Output trigers: {"EMC", "CELLSTRGR", 0, Lifetime::Timeframe}
  /// Output MC-truth: {"EMC", "CELLSMCTR", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 protected:
  std::vector<o2::emcal::SRUBunchContainer> digitsToBunches(gsl::span<const o2::emcal::Digit> digits, std::vector<gsl::span<const o2::emcal::MCLabel>>& mcLabels);

  std::vector<AltroBunch> findBunches(const std::vector<const o2::emcal::Digit*>& channelDigits, const std::vector<gsl::span<const o2::emcal::MCLabel>>& mcLabels, ChannelType_t channelType);

  void mergeLabels(std::vector<o2::emcal::AltroBunch>& channelBunches);

  int selectMaximumBunch(const gsl::span<const Bunch>& bunchvector);

 private:
  bool mPropagateMC = false;                                           ///< Switch whether to process MC true labels
  bool mLoadRecoParamFromCCDB = false;                                 ///< Flag to load the the SimParams from CCDB
  unsigned int mSubspecificationIn = 0;                                ///< Input subspecification
  unsigned int mSubspecificationOut = 0;                               ///< Output subspecification
  o2::emcal::Geometry* mGeometry = nullptr;                            ///!<! Geometry pointer
  std::unique_ptr<o2::emcal::CaloRawFitter> mRawFitter;                ///!<! Raw fitter
  std::vector<o2::emcal::Cell> mOutputCells;                           ///< Container with output cells
  std::vector<o2::emcal::TriggerRecord> mOutputTriggers;               ///< Container with output trigger records
  o2::dataformats::MCTruthContainer<o2::emcal::MCLabel> mOutputLabels; ///< Container with output MC labels
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
/// \ingroup EMCALworkflow
/// \param propagateMC If true the MC truth container is propagated to the output
/// \param useccdb If true the RecoParams are loaded from the CCDB
/// \param inputSubsepc Subspec of input objects
/// \param outputSubspec Subspec of output objects
///
/// Refer to CellConverterSpec::run for input and output specs
framework::DataProcessorSpec getCellConverterSpec(bool propagateMC, bool useccdb = false, int inputSubsepc = 0, int outputSubspec = 0);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2

#endif // O2_EMCAL_CELLCONVERTER_SPEC
