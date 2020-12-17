// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Table definitions for EMCAL analysis clusters
//
// Author: Raymond Ehlers

#pragma once

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace emcalcluster
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(Energy, energy, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(M02, m02, float);
} // namespace emcalcluster

DECLARE_SOA_TABLE(EMCALClusters, "AOD", "EMCALCLUSTERS",
                  o2::soa::Index<>, emcalcluster::CollisionId, emcalcluster::Energy,
                  emcalcluster::Eta, emcalcluster::Phi, emcalcluster::M02);

using EMCALCluster = EMCALClusters::iterator;

} // namespace o2::aod
