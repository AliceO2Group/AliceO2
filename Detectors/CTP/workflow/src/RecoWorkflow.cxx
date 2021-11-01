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

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "FairLogger.h"

#include "Framework/RootSerializationSupport.h"
#include "Algorithm/RangeTokenizer.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsCTP/Digits.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CTPWorkflow/RecoWorkflow.h"
#include "CTPWorkflowIO/DigitReaderSpec.h"
#include "CTPWorkflowIO/DigitWriterSpec.h"
#include "CTPWorkflow/RawToDigitConverterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::dataformats;

namespace o2
{

namespace ctp
{

namespace reco_workflow
{

o2::framework::WorkflowSpec getWorkflow(bool noLostTF)
{
  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::ctp::reco_workflow::getRawToDigitConverterSpec(noLostTF));
  return std::move(specs);
}

} // namespace reco_workflow

} // namespace ctp

} // namespace o2
