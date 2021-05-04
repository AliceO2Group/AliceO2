// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_CLUSTERREADERSPEC_H
#define O2_TPC_CLUSTERREADERSPEC_H
/// @file   ClusterReaderSpec.h
/// @author David Rohr

#include "Framework/WorkflowSpec.h"
#include <vector>

namespace o2
{
namespace tpc
{
framework::DataProcessorSpec getClusterReaderSpec(bool useMC, const std::vector<int>* tpcSectors = nullptr, const std::vector<int>* laneConfiguration = nullptr);
} // end namespace tpc
} // end namespace o2
#endif // O2_TPC_CLUSTERREADERSPEC_H
