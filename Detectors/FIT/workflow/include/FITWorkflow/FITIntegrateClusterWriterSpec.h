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

#ifndef O2_FIT_FITINTEGRATECLUSTERWRITER_SPEC
#define O2_FIT_FITINTEGRATECLUSTERWRITER_SPEC

#include <vector>
#include <boost/algorithm/string.hpp>
#include "FITWorkflow/FITIntegrateClusterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace fit
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

template <typename DataT>
DataProcessorSpec getFITIntegrateClusterWriterSpec()
{
  using FitType = DataDescriptionFITCurrents<DataT>;

  const std::string treeFile = fmt::format("o2currents_{}.root", FitType::getName());
  const std::string treeName = FitType::getName();
  return MakeRootTreeWriterSpec(fmt::format("{}-currents-writer", FitType::getName()).data(),
                                treeFile.data(),
                                treeName.data(),
                                BranchDefinition<typename DataDescriptionFITCurrents<DataT>::DataTStruct>{InputSpec{"ifitc", FitType::getDataOrigin(), FitType::getDataDescriptionFITC(), 0}, "IFITC", 1},
                                BranchDefinition<o2::dataformats::TFIDInfo>{InputSpec{"tfID", FitType::getDataOrigin(), FitType::getDataDescriptionFITTFId(), 0}, "tfID", 1})();
}

} // end namespace fit
} // end namespace o2

#endif
