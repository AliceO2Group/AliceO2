// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/DigitsToRawWorkflow.cxx
/// \brief  Definition of MID reconstruction workflow for MC
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 September 2019

#include "MIDWorkflow/DigitsToRawWorkflow.h"

#include "DPLUtils/Utils.h"
#include "MIDWorkflow/DigitReaderSpec.h"
#include "MIDWorkflow/RawWriterSpec.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

of::WorkflowSpec getDigitsToRawWorkflow()
{

  auto checkReady = [](o2::framework::DataRef const& ref) {
    // The default checkReady function is not defined in MakeRootTreeWriterSpec:
    // this leads to a std::exception in DPL.
    // While the exception seems harmless (the processing goes on without apparent consequence),
    // it is quite ugly to see.
    // So, let us define checkReady here.
    // FIXME: to be fixed in MakeRootTreeWriterSpec
    return false;
  };

  of::WorkflowSpec specs;

  specs.emplace_back(getDigitReaderSpec());
  specs.emplace_back(getRawWriterSpec());
  return specs;
}
} // namespace mid
} // namespace o2
