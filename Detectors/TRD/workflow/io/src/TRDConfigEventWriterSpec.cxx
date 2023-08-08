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

#ifndef CONFIGEVENT_SRC_TRDDIGITWRITERSPEC_H_
#define CONFIGEVENT_SRC_TRDDIGITWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "DataFormatsTRD/Digit.h"
#include "TRDWorkflowIO/TRDConfigEventWriterSpec.h"

namespace o2
{
namespace trd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getTRDConfigEventWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  using DataRef = framework::DataRef;
  LOGP(info, " ZZZ writing config event to root file");

  return MakeRootTreeWriterSpec("TRDConfigEventWriter",
                                "trdconfigevents.root",
                                "o2sim",
                                // setting a custom callback for closing the writer
                                BranchDefinition<o2::trd::TrapConfigEvent>{InputSpec{"trapconfig", o2::header::gDataOriginTRD, "TRDCFG"}, "ConfigEvent"})();
}

} // end namespace trd
} // end namespace o2

#endif /* CONFIGEVENT_SRC_TRDDIGITWRITERSPEC_H_ */
