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

/// @file   RecoWorkflow.cxx

#include "FDDWorkflow/RecoWorkflow.h"

#include "FDDWorkflow/DigitReaderSpec.h"
#include "FDDWorkflow/RecPointWriterSpec.h"
#include "FDDWorkflow/ReconstructorSpec.h"

namespace o2
{
namespace fdd
{

framework::WorkflowSpec getRecoWorkflow(bool useMC, bool disableRootInp, bool disableRootOut)
{
  framework::WorkflowSpec specs;

  if (!disableRootInp) {
    specs.emplace_back(o2::fdd::getFDDRecPointWriterSpec(useMC));
  }
  specs.emplace_back(o2::fdd::getFDDReconstructorSpec(useMC));
  if (!disableRootOut) {
    specs.emplace_back(o2::fdd::getFDDDigitReaderSpec(useMC));
  }

  return specs;
}

} // namespace fdd
} // namespace o2
