// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

framework::WorkflowSpec getRecoWorkflow(bool useMC)
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::fdd::getFDDRecPointWriterSpec(useMC));
  specs.emplace_back(o2::fdd::getFDDReconstructorSpec(useMC));
  specs.emplace_back(o2::fdd::getFDDDigitReaderSpec(useMC));

  return specs;
}

} // namespace fdd
} // namespace o2
