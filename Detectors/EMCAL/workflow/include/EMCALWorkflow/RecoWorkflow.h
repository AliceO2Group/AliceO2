// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
enum struct InputType { Digitizer, ///< directly read digits from channel {TPC:DIGITS)
                        Digits,    ///< read digits from file
                        Cells,     ///< read compressed cells from file
                        Raw,       ///< read data in raw page format from file
                        Clusters   ///< read native clusters from file
};

/// \enum OutputType
/// \brief Output types of the workflow
/// \ingroup EMCALworkflow
enum struct OutputType { Digits,  ///< EMCAL digits
                         Cells,   ///< EMCAL cells
                         Raw,     ///< EMCAL raw data
                         Clusters ///< EMCAL clusters
};

/// \brief create the workflow for EMCAL reconstruction
/// \param propagateMC If true MC labels are propagated to the output files
/// \param enableDigitsPrinter If true
/// \param cfgInput Input objects processed in the workflow
/// \param cfgOutput Output objects created in the workflow
/// \return EMCAL reconstruction workflow for the configuration provided
/// \ingroup EMCALwokflow
framework::WorkflowSpec getWorkflow(bool propagateMC = true,
                                    bool enableDigitsPrinter = false,
                                    std::string const& cfgInput = "digits",   //
                                    std::string const& cfgOutput = "clusters" //
);
} // namespace reco_workflow

} // namespace emcal

} // namespace o2
#endif
