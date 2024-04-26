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

/// \file MilleRecordWriterSpec.cxx
/// \brief Implementation of a data processor to write MillePede record in a root file
///
/// \author Chi Zhang, CEA-Saclay, chi.zhang@cern.ch

#include "ForwardAlign/MilleRecordWriterSpec.h"

#include <vector>
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "ForwardAlign/MillePedeRecord.h"

namespace o2
{
namespace fwdalign
{

using namespace o2::framework;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getMilleRecordWriterSpec(bool useMC, const char* specName, const char* fileName)
{
  return MakeRootTreeWriterSpec(specName,
                                fileName,
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree MillePede records for MCH-MID tracks"},
                                BranchDefinition<fwdalign::MillePedeRecord>{InputSpec{"data", "MUON", "RECORD_MCHMID", Lifetime::Sporadic}, "data"})();
}

} // namespace fwdalign
} // namespace o2