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
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/Multiplicity.h"

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

namespace o2::aod
{

namespace femtodreamcollision
{
DECLARE_SOA_COLUMN(MultV0M, multV0M, float);
DECLARE_SOA_COLUMN(Sphericity, sphericity, float);
} // namespace femtodreamcollision
DECLARE_SOA_TABLE(FemtoDreamCollisions, "AOD", "FemtoDreamCols",
                  o2::soa::Index<>,
                  o2::aod::collision::PosZ,
                  femtodreamcollision::MultV0M,
                  femtodreamcollision::Sphericity);
using FemtoDreamCollision = FemtoDreamCollisions::iterator;

namespace femtodreamparticle
{
DECLARE_SOA_INDEX_COLUMN(FemtoDreamCollision, femtoDreamCollision);
DECLARE_SOA_COLUMN(pT, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Charge, charge, uint8_t);
//DECLARE_SOA_COLUMN(PartSel, partSel, uint8_t);
} // namespace femtodreamparticle
DECLARE_SOA_TABLE(FemtoDreamParticles, "AOD", "FemtoDreamParts",
                  o2::soa::Index<>,
                  femtodreamparticle::FemtoDreamCollisionId,
                  femtodreamparticle::pT,
                  femtodreamparticle::Eta,
                  femtodreamparticle::Phi,
                  femtodreamparticle::Charge);
using FemtoDreamParticle = FemtoDreamParticles::iterator;

namespace hash
{
DECLARE_SOA_COLUMN(Bin, bin, int);
} // namespace hash
DECLARE_SOA_TABLE(Hashes, "AOD", "HASH", hash::Bin);
using Hash = Hashes::iterator;

namespace mixingEvent
{
DECLARE_SOA_COLUMN(Use, use, int);
} // namespace hash
DECLARE_SOA_TABLE(MixingEvents, "AOD", "ME", mixingEvent::Use);
using MixingEvent = MixingEvents::iterator;


} // namespace o2::aod

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODERIVED_H_ */
