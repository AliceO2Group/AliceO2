// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RECOWORKFLOW_H
#define O2_TPC_RECOWORKFLOW_H
/// @file   RecoWorkflow.h
/// @author Matthias Richter
/// @since  2018-09-26
/// @brief  Workflow definition for the TPC reconstruction

#include "Framework/WorkflowSpec.h"
#include <string>
namespace o2
{
namespace TPC
{

namespace RecoWorkflow
{
/// define input and output types of the workflow
enum struct InputType { Digitizer, // directly read digits from  {TPC:DIGITS}
                        Digits,    // read digits from file
                        Clusters,  // read clusters from file
                        Raw };
enum struct OutputType { Clusters,
                         Raw,
                         DecodedClusters,
                         Tracks };

/// create the workflow for TPC reconstruction
framework::WorkflowSpec getWorkflow(bool propagateMC = true, int nLanes = 1,                               //
                                    std::string inputType = "digitizer", std::string outputType = "tracks" //
                                    );

} // end namespace RecoWorkflow
} // end namespace TPC
} // end namespace o2
#endif //O2_TPC_RECOWORKFLOW_H
