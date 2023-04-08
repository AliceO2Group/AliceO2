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

#include "MCHWorkflow/ErrorWriterSpec.h"

#include <vector>
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "MCHBase/Error.h"

using namespace o2::framework;

namespace o2::mch
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getErrorWriterSpec(const char* specName)
{
  return MakeRootTreeWriterSpec(specName,
                                "mcherrors.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree MCH Errors"},
                                BranchDefinition<std::vector<Error>>{InputSpec{"errors", "MCH", "PROCERRORS"}, "errors"})();
}

} // namespace o2::mch
