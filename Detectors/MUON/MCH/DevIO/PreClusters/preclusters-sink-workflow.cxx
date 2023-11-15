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

/// \file   preclusters-sink-workflow.cxx
/// \author Philippe Pillot, Subatech
/// \author  Andrea Ferrero
///
/// \brief This is an executable that dumps to a file on disk the preclusters received via DPL.
///
/// This is an executable that dumps to a file on disk the preclusters received via the Data Processing Layer.
/// It can be used to debug the preclustering step. For example, one can do:
/// \code{.sh}
/// o2-mch-file-to-digits-workflow --infile=some_data_file | o2-mch-digits-to-preclusters-workflow | o2-mch-preclusters-sink-workflow --outfile preclusters.bin
/// \endcode
///

#include "Framework/runDataProcessing.h"

#include "PreClusterSinkSpec.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  return WorkflowSpec{o2::mch::getPreClusterSinkSpec()};
}
