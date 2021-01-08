// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace phos
{

namespace reco_workflow
{

/// \class DigitsPrinterSpec
/// \brief Example task for PHOS digits monitoring
/// \author Dmitri Peresunko after Markus Fasel
/// \since Dec 14, 2019
///
/// Example payload for workflows using o2::phos::Digit as payload.
/// Printing several digit-related information for each digit. Refer
/// to run for the list of input spec to be specified.
class DigitsPrinterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  DigitsPrinterSpec() = default;

  /// \brief Destructor
  ~DigitsPrinterSpec() override = default;

  /// \brief Initializing the digits printer task
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Printing digit-related information
  /// \param ctx Processing context
  ///
  /// Printing energy and absID for each digit.
  /// Following input branches are linked:
  /// - digits: {"PHS", "DIGITS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;
};

/// \brief Creating digits printer spec
///
/// Refer to DigitsPrinterSpec::run for a list of input
/// specs
o2::framework::DataProcessorSpec getPhosDigitsPrinterSpec();
} // namespace reco_workflow
} // namespace phos

} // namespace o2
