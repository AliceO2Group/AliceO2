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

#ifndef STEER_DIGITIZERWORKFLOW_SRC_MIDDIGITWRITERSPEC_H
#define STEER_DIGITIZERWORKFLOW_SRC_MIDDIGITWRITERSPEC_H

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/MCLabel.h"

namespace o2
{
namespace mid
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getMIDDigitWriterSpec(bool mctruth)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("MIDDigitWriter",
                                "middigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<ColumnData>>{InputSpec{"middigits", "MID", "DIGITS"}, "MIDDigit"},
                                BranchDefinition<std::vector<ROFRecord>>{InputSpec{"midrofrecords", "MID", "DIGITSROF"}, "MIDROFRecords"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<MCLabel>>{InputSpec{"middigitlabels", "MID", "DIGITLABELS"}, "MIDDigitMCLabels", mctruth ? 1 : 0})();
}

} // namespace mid
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_MIDDIGITWRITERSPEC_H */
