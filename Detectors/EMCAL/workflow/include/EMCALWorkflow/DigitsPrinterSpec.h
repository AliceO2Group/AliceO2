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

namespace emcal
{

namespace reco_workflow
{

/// \class DigitsPrinterSpec
/// \brief Example task for EMCAL digits monitoring
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since July 11, 2019
///
/// Example payload for workflows using o2::emcal::Digit as payload.
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
  /// Printing energy and tower ID for each digit.
  /// Following input branches are linked:
  /// - digits: {"EMC", "DIGITS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;
};

/// \brief Creating digits printer spec
///
/// Refer to DigitsPrinterSpec::run for a list of input
/// specs
o2::framework::DataProcessorSpec getEmcalDigitsPrinterSpec();
} // namespace reco_workflow
} // namespace emcal

} // namespace o2
