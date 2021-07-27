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

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_

#include "Framework/ASoA.h"
#include "MathUtils/Utils.h"
#include "Framework/DataTypes.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/Expressions.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include <cmath>

namespace o2::aod
{

/// FemtoDreamCollision
namespace femtodreamcollision
{
DECLARE_SOA_COLUMN(MultV0M, multV0M, float);       //! V0M multiplicity
DECLARE_SOA_COLUMN(Sphericity, sphericity, float); //! Sphericity of the event
} // namespace femtodreamcollision

DECLARE_SOA_TABLE(FemtoDreamCollisions, "AOD", "FEMTODREAMCOLS",
                  o2::soa::Index<>,
                  o2::aod::collision::PosZ,
                  femtodreamcollision::MultV0M,
                  femtodreamcollision::Sphericity);
using FemtoDreamCollision = FemtoDreamCollisions::iterator;

/// FemtoDreamTrack
namespace femtodreamparticle
{
/// Distinuishes the different particle types
enum ParticleType {
  kTrack,          //! Track
  kV0,             //! V0
  kV0Child,        //! Child track of a V0
  kCascade,        //! Cascade
  kCascadeBachelor //! Bachelor track of a cascade
};

static constexpr std::string_view ParticleTypeName[5] = {"Tracks", "V0", "V0Child", "Cascade", "CascadeBachelor"}; //! Naming of the different particle types

using cutContainerType = uint32_t; //! Definition of the data type for the bit-wise container for the different selection criteria

DECLARE_SOA_INDEX_COLUMN(FemtoDreamCollision, femtoDreamCollision);
DECLARE_SOA_COLUMN(Pt, pt, float);                    //! p_T (GeV/c)
DECLARE_SOA_COLUMN(Eta, eta, float);                  //! Eta
DECLARE_SOA_COLUMN(Phi, phi, float);                  //! Phi
DECLARE_SOA_COLUMN(PartType, partType, uint8_t);      //! Type of the particle, according to femtodreamparticle::ParticleType
DECLARE_SOA_COLUMN(Cut, cut, cutContainerType);       //! Bit-wise container for the different selection criteria
DECLARE_SOA_COLUMN(PIDCut, pidcut, cutContainerType); //! Bit-wise container for the different PID selection criteria \todo since bit-masking cannot be done yet with filters we use a second field for the PID
DECLARE_SOA_COLUMN(TempFitVar, tempFitVar, float);    //! Observable for the template fitting (Track: DCA_xy, V0: CPA)
DECLARE_SOA_COLUMN(Indices, indices, int[2]);         //! Field for the track indices to remove auto-correlations

DECLARE_SOA_DYNAMIC_COLUMN(Theta, theta, //! Compute the theta of the track
                           [](float eta) -> float {
                             return 2.f * std::atan(std::exp(-eta));
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, //! Compute the momentum in x in GeV/c
                           [](float pt, float phi) -> float {
                             return pt * std::sin(phi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //! Compute the momentum in y in GeV/c
                           [](float pt, float phi) -> float {
                             return pt * std::cos(phi);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //! Compute the momentum in z in GeV/c
                           [](float pt, float eta) -> float {
                             return pt * std::sinh(eta);
                           });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //! Compute the overall momentum in GeV/c
                           [](float pt, float eta) -> float {
                             return pt * std::cosh(eta);
                           });
// debug variables
DECLARE_SOA_COLUMN(Sign, sign, int8_t);                                                  //! Sign of the track charge
DECLARE_SOA_COLUMN(TPCNClsFound, tpcNClsFound, uint8_t);                                 //! Number of TPC clusters
DECLARE_SOA_COLUMN(TPCNClsCrossedRows, tpcNClsCrossedRows, uint8_t);                     //! Number of TPC crossed rows
DECLARE_SOA_DYNAMIC_COLUMN(TPCCrossedRowsOverFindableCls, tpcCrossedRowsOverFindableCls, //! Compute the number of crossed rows over findable TPC clusters
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
                  femtodreamparticle::PartType,
                  femtodreamparticle::Cut,
                  femtodreamparticle::PIDCut,
                  femtodreamparticle::TempFitVar,
                  femtodreamparticle::Indices,
                  femtodreamparticle::Theta<femtodreamparticle::Eta>,
                  femtodreamparticle::Px<femtodreamparticle::Pt, femtodreamparticle::Phi>,
                  femtodreamparticle::Py<femtodreamparticle::Pt, femtodreamparticle::Phi>,
                  femtodreamparticle::Pz<femtodreamparticle::Pt, femtodreamparticle::Eta>,
                  femtodreamparticle::P<femtodreamparticle::Pt, femtodreamparticle::Eta>);
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

/// Hash
namespace hash
{
DECLARE_SOA_COLUMN(Bin, bin, int); //! Hash for the event mixing
} // namespace hash
DECLARE_SOA_TABLE(Hashes, "AOD", "HASH", hash::Bin);
using Hash = Hashes::iterator;

} // namespace o2::aod

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_ */
