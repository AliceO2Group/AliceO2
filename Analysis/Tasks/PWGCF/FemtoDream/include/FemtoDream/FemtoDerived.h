// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_

#include "Framework/ASoA.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"

namespace o2::aod
{

namespace femtodreamcollision
{
DECLARE_SOA_COLUMN(MultV0M, multV0M, float);
DECLARE_SOA_COLUMN(Sphericity, sphericity, float);
} // namespace femtodreamcollision
DECLARE_SOA_TABLE(FemtoDreamCollisions, "AOD", "FEMTODREAMCOLS",
                  o2::soa::Index<>,
                  o2::aod::collision::PosZ,
                  femtodreamcollision::MultV0M,
                  femtodreamcollision::Sphericity);
using FemtoDreamCollision = FemtoDreamCollisions::iterator;

namespace femtodreamparticle
{
DECLARE_SOA_INDEX_COLUMN(FemtoDreamCollision, femtoDreamCollision);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Cut, cut, uint64_t);
DECLARE_SOA_COLUMN(TempFitVar, tempFitVar, float);
// debug variables
DECLARE_SOA_COLUMN(Sign, sign, int8_t);
DECLARE_SOA_COLUMN(TPCNClsFound, tpcNClsFound, uint8_t);
DECLARE_SOA_COLUMN(TPCNClsCrossedRows, tpcNClsCrossedRows, uint8_t);
DECLARE_SOA_DYNAMIC_COLUMN(TPCCrossedRowsOverFindableCls, tpcCrossedRowsOverFindableCls, //!
                           [](int8_t tpcNClsFindable, int8_t tpcNClsCrossedRows) -> float {
                             return (float)tpcNClsCrossedRows / (float)tpcNClsFindable;
                           });

} // namespace femtodreamparticle
DECLARE_SOA_TABLE(FemtoDreamParticles, "AOD", "FEMTODREAMPARTS",
                  o2::soa::Index<>,
                  femtodreamparticle::FemtoDreamCollisionId,
                  femtodreamparticle::Pt,
                  femtodreamparticle::Eta,
                  femtodreamparticle::Phi,
                  femtodreamparticle::Cut,
                  femtodreamparticle::TempFitVar);
using FemtoDreamParticle = FemtoDreamParticles::iterator;

DECLARE_SOA_TABLE(FemtoDreamDebugParticles, "AOD", "FEMTODEBUGPARTS",
                  soa::Index<>,
                  femtodreamparticle::FemtoDreamCollisionId,
                  femtodreamparticle::Sign,
                  femtodreamparticle::TPCNClsFound,
                  track::TPCNClsFindable,
                  femtodreamparticle::TPCNClsCrossedRows,
                  track::TPCNClsShared,
                  femtodreamparticle::TPCCrossedRowsOverFindableCls<track::TPCNClsFindable, femtodreamparticle::TPCNClsCrossedRows>,
                  track::DcaXY,
                  track::DcaZ,
                  pidtpc_tiny::TPCNSigmaStoreEl,
                  pidtpc_tiny::TPCNSigmaStorePi,
                  pidtpc_tiny::TPCNSigmaStoreKa,
                  pidtpc_tiny::TPCNSigmaStorePr,
                  pidtpc_tiny::TPCNSigmaStoreDe,
                  pidtpc_tiny::TPCNSigmaEl<pidtpc_tiny::TPCNSigmaStoreEl>,
                  pidtpc_tiny::TPCNSigmaPi<pidtpc_tiny::TPCNSigmaStorePi>,
                  pidtpc_tiny::TPCNSigmaKa<pidtpc_tiny::TPCNSigmaStoreKa>,
                  pidtpc_tiny::TPCNSigmaPr<pidtpc_tiny::TPCNSigmaStorePr>,
                  pidtpc_tiny::TPCNSigmaDe<pidtpc_tiny::TPCNSigmaStoreDe>,
                  pidtof_tiny::TOFNSigmaStoreEl,
                  pidtof_tiny::TOFNSigmaStorePi,
                  pidtof_tiny::TOFNSigmaStoreKa,
                  pidtof_tiny::TOFNSigmaStorePr,
                  pidtof_tiny::TOFNSigmaStoreDe,
                  pidtof_tiny::TOFNSigmaEl<pidtof_tiny::TOFNSigmaStoreEl>,
                  pidtof_tiny::TOFNSigmaPi<pidtof_tiny::TOFNSigmaStorePi>,
                  pidtof_tiny::TOFNSigmaKa<pidtof_tiny::TOFNSigmaStoreKa>,
                  pidtof_tiny::TOFNSigmaPr<pidtof_tiny::TOFNSigmaStorePr>,
                  pidtof_tiny::TOFNSigmaDe<pidtof_tiny::TOFNSigmaStoreDe>);
using FemtoDreamDebugParticle = FemtoDreamDebugParticles::iterator;

} // namespace o2::aod

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_ */
