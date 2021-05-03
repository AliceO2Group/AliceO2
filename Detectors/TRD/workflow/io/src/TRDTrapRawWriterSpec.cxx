// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDTRAPSIMULATORTRAPRAWWRITER_H
#define O2_TRDTRAPSIMULATORTRAPRAWWRITER_H

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "TRDWorkflowIO/TRDTrapRawWriterSpec.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/LinkRecord.h"

#include <fstream>
#include <iostream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getTRDTrapRawWriterSpec()
{
  //  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("TRDTrkltRawWrt",
                                "trdtrapraw.root",
                                "o2sim",
                                BranchDefinition<std::vector<uint32_t>>{InputSpec{"trapraw", "TRD", "RAWDATA"}, "TrapRaw"},
                                BranchDefinition<std::vector<o2::trd::LinkRecord>>{InputSpec{"traplinks", "TRD", "RAWLNKRD"}, "TrapLinkRecord"},
                                BranchDefinition<std::vector<o2::trd::TriggerRecord>>{InputSpec{"traprawtrigrec", "TRD", "RAWTRGRD"}, "RawTriggerRecord"})();
};

} // end namespace trd
} // end namespace o2

#endif // O2_TRDTRAPSIMULATORTRACKLETWRITER_H
