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

/// @brief  Processor spec for a ROOT file writer for EMCAL digits

#include "EMCALDigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include <SimulationDataFormat/MCTruthContainer.h>
#include "DataFormatsEMCAL/Digit.h"
#include <DataFormatsEMCAL/MCLabel.h>
#include "DataFormatsEMCAL/TriggerRecord.h"

using namespace o2::framework;

namespace o2
{
namespace emcal
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

/// create the processor spec
/// describing a processor receiving digits for EMCal writing them to file
DataProcessorSpec getEMCALDigitWriterSpec(bool mctruth)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("EMCALDigitWriter",
                                "emcaldigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::emcal::Digit>>{InputSpec{"emcaldigits", "EMC", "DIGITS"}, "EMCALDigit"},
                                BranchDefinition<std::vector<o2::emcal::TriggerRecord>>{InputSpec{"trgrecorddigits", "EMC", "TRGRDIG"}, "EMCALDigitTRGR"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>>{InputSpec{"emcaldigitlabels", "EMC", "DIGITSMCTR"}, "EMCALDigitMCTruth", mctruth ? 1 : 0})();
}
} // end namespace emcal
} // end namespace o2
