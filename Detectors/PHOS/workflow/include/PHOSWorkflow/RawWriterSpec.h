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
#include "PHOSSimulation/RawWriter.h"
#include "Framework/Task.h"

namespace o2
{

namespace phos
{

namespace reco_workflow
{

/// \class RawWriterSpec
/// \brief Coverter task for PHOS digits to raw
/// \author Dmitri Peresunko after Markus Fasel
/// \since Nov.20, 2020
///
/// Task converting a vector of PHOS digits to raw file
class RawWriterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawWriterSpec() : framework::Task(){};

  /// \brief Destructor
  ~RawWriterSpec() override = default;

  /// \brief Initializing the RawWriterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of digits to cells
  /// \param ctx Processing context
  ///
  /// Converting the input vector of o2::phos::Digit to
  /// file with raw data
  ///
  /// The following branches are linked:
  /// Input digits: {"PHS", "DIGITS", 0, Lifetime::Timeframe}
  /// Input digit trigger records: {"PHS", "DIGITTRS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  o2::phos::RawWriter* mRawWriter = nullptr;
};

/// \brief Creating DataProcessorSpec for the PHOS Cell Converter Spec
/// \param propagateMC If true the MC truth container is propagated to the output
///
/// Refer to RawWriterSpec::run for input and output specs
framework::DataProcessorSpec getRawWriterSpec();

} // namespace reco_workflow

} // namespace phos

} // namespace o2
