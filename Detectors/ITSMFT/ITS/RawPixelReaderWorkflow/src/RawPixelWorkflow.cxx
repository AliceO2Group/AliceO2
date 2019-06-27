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

#include "ITSRawWorkflow/RawPixelWorkflow.h"
#include "ITSRawWorkflow/RawPixelReaderSpec.h"
#include "ITSRawWorkflow/RawPixelGetterSpec.h"

namespace o2
{
namespace its
{

namespace RawPixelWorkflow
{

framework::WorkflowSpec getWorkflow()
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::its::getRawPixelReaderSpec());
  specs.emplace_back(o2::its::getRawPixelGetterSpec());

  return specs;
}

} // namespace RawPixelWorkflow

} // namespace its
} // namespace o2
