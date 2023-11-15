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

/// @file   ClusterWriterWorkflow.cxx

#include "ITS3Workflow/ClusterWriterWorkflow.h"
#include "ITS3Workflow/ClusterWriterSpec.h"

namespace o2
{
namespace its3
{

namespace cluster_writer_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC)
{
  framework::WorkflowSpec specs;

  specs.emplace_back(getClusterWriterSpec(useMC));

  return specs;
}

} // namespace cluster_writer_workflow
} // namespace its3
} // namespace o2
