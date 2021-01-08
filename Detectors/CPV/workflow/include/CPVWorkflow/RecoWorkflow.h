// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CPV_RECOWORKFLOW_H
#define O2_CPV_RECOWORKFLOW_H

#include "Framework/WorkflowSpec.h"
#include <string>
#include <vector>

namespace o2
{

namespace cpv
{

namespace reco_workflow
{

/// define input and output types of the workflow
enum struct InputType { Hits,   // read hits from file
                        Digits, // read digits from file
                        Raw     // read data in raw page format from file
};
enum struct OutputType { Digits,
                         Raw,
                         Clusters
};

/// create the workflow for CPV reconstruction
framework::WorkflowSpec getWorkflow(bool propagateMC = true,
                                    bool enableDigitsPrinter = false,
                                    std::string const& cfgInput = "digits",   //
                                    std::string const& cfgOutput = "clusters" //
);
} // namespace reco_workflow

} // namespace cpv

} // namespace o2
#endif
