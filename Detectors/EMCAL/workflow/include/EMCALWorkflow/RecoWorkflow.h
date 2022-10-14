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

#ifndef O2_EMCAL_RECOWORKFLOW_H
#define O2_EMCAL_RECOWORKFLOW_H

#include "Framework/WorkflowSpec.h"
#include <string>
#include <vector>

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \enum InputType
/// \brief Input types of the workflow
/// \ingroup EMCALworkflow
enum struct InputType { Digits,  ///< read digits from file
                        Cells,   ///< read compressed cells from file
                        Raw,     ///< read data in raw page format from file
                        Clusters ///< read native clusters from file
};

/// \enum OutputType
/// \brief Output types of the workflow
/// \ingroup EMCALworkflow
enum struct OutputType { Digits,          ///< EMCAL digits
                         Cells,           ///< EMCAL cells
                         Raw,             ///< EMCAL raw data
                         Clusters,        ///< EMCAL clusters
                         AnalysisClusters ///< EMCAL analysis clusters
};

/// \brief create the workflow for EMCAL reconstruction
/// \param propagateMC If true MC labels are propagated to the output files
/// \param askDISTSTF If true the Raw->Cell converter subscribes to FLP/DISTSUBTIMEFRAME
/// \param enableDigitsPrinter If true then the simple digits printer is added as dummy task
/// \param subspecification Subspecification in case of running on different FLPs
/// \param cfgInput Input objects processed in the workflow
/// \param cfgOutput Output objects created in the workflow
/// \param loadRecoParamsFromCCDB Load the reco params from the CCDB
/// \return EMCAL reconstruction workflow for the configuration provided
/// \ingroup EMCALwokflow
framework::WorkflowSpec getWorkflow(bool propagateMC = true,
                                    bool askDISTSTF = true,
                                    bool enableDigitsPrinter = false,
                                    int subspecification = 0,
                                    std::string const& cfgInput = "digits",
                                    std::string const& cfgOutput = "clusters",
                                    bool disableRootInput = false,
                                    bool disableRootOutput = false,
                                    bool disableDecodingErrors = false);
} // namespace reco_workflow

} // namespace emcal

} // namespace o2
#endif
