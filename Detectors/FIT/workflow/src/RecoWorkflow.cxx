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

#include "FITWorkflow/RecoWorkflow.h"

#include "FITWorkflow/FT0DigitReaderSpec.h"
#include "FITWorkflow/FT0RecPointWriterSpec.h"
#include "FITWorkflow/FT0ReconstructorSpec.h"

namespace o2
{
namespace fit
{

framework::WorkflowSpec getRecoWorkflow(bool useMC)
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::ft0::getFT0RecPointWriterSpec(useMC));
  specs.emplace_back(o2::ft0::getFT0ReconstructorSpec(useMC));
  specs.emplace_back(o2::ft0::getFT0DigitReaderSpec(useMC));

  return specs;
}

} // namespace fit
} // namespace o2
