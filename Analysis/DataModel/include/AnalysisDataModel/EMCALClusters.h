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

// Table definitions for EMCAL analysis clusters
//
// Author: Raymond Ehlers

#ifndef O2_ANALYSIS_DATAMODEL_EMCALCLUSTERS
#define O2_ANALYSIS_DATAMODEL_EMCALCLUSTERS

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace emcalcluster
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision); //!
DECLARE_SOA_COLUMN(Energy, energy, float);      //!
DECLARE_SOA_COLUMN(Eta, eta, float);            //!
DECLARE_SOA_COLUMN(Phi, phi, float);            //!
DECLARE_SOA_COLUMN(M02, m02, float);            //!
} // namespace emcalcluster

DECLARE_SOA_TABLE(EMCALClusters, "AOD", "EMCALCLUSTERS", //!
                  o2::soa::Index<>, emcalcluster::CollisionId,
                  emcalcluster::Energy, emcalcluster::Eta, emcalcluster::Phi,
                  emcalcluster::M02);

using EMCALCluster = EMCALClusters::iterator;

} // namespace o2::aod

#endif
