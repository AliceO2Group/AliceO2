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

// table definitions for jets
//
// Author: Jochen Klein, Nima Zardoshti

#ifndef O2_ANALYSIS_DATAMODEL_JET_H
#define O2_ANALYSIS_DATAMODEL_JET_H

#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/EMCALClusters.h"
#include <cmath>

namespace o2::aod
{
namespace jet
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision); //!
DECLARE_SOA_COLUMN(Pt, pt, float);              //!
DECLARE_SOA_COLUMN(Eta, eta, float);            //!
DECLARE_SOA_COLUMN(Phi, phi, float);            //!
DECLARE_SOA_COLUMN(Energy, energy, float);      //!
DECLARE_SOA_COLUMN(Mass, mass, float);          //!
DECLARE_SOA_COLUMN(Area, area, float);          //!
DECLARE_SOA_COLUMN(MatchedJetIndex, matchedJetIndex, int);  //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,              //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //! absolute p
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace jet

DECLARE_SOA_TABLE(Jets, "AOD", "JET", //!
                  o2::soa::Index<>,
                  jet::CollisionId,
                  jet::Pt,
                  jet::Eta,
                  jet::Phi,
                  jet::Energy,
                  jet::Mass,
                  jet::Area,
                  jet::MatchedJetIndex,
                  jet::Px<jet::Pt, jet::Phi>,
                  jet::Py<jet::Pt, jet::Phi>,
                  jet::Pz<jet::Pt, jet::Eta>,
                  jet::P<jet::Pt, jet::Eta>);

using Jet = Jets::iterator;

DECLARE_SOA_EXTENDED_TABLE(JetsMCParticleLevel, Jets, "JETMCPART"); //!
DECLARE_SOA_EXTENDED_TABLE(JetsMCDetectorLevel, Jets, "JETMCDET"); //!
DECLARE_SOA_EXTENDED_TABLE(JetsHybridIntermediate, Jets, "JETHYBRDIINTERM"); //!

// TODO: absorb in jet table
// when list of references available
namespace constituents
{
DECLARE_SOA_INDEX_COLUMN(Jet, jet);     //!
DECLARE_SOA_INDEX_COLUMN(Track, track); //!
DECLARE_SOA_INDEX_COLUMN(EMCALCluster, cluster); //!
} // namespace constituents

DECLARE_SOA_TABLE(JetTrackConstituents, "AOD", "TRKCONSTITS", //!
                  constituents::JetId,
                  constituents::TrackId);
DECLARE_SOA_TABLE(JetClusterConstituents, "AOD", "CLUSCONSTITS", //!
                  constituents::JetId,
                  constituents::EMCALClusterId);

using JetTrackConstituent = JetTrackConstituents::iterator;
using JetClusterConstituent = JetClusterConstituents::iterator;

namespace constituentssub
{
DECLARE_SOA_INDEX_COLUMN(Jet, jet);        //!
DECLARE_SOA_COLUMN(Pt, pt, float);         //!
DECLARE_SOA_COLUMN(Eta, eta, float);       //!
DECLARE_SOA_COLUMN(Phi, phi, float);       //!
DECLARE_SOA_COLUMN(Energy, energy, float); //!
DECLARE_SOA_COLUMN(Mass, mass, float);     //!
DECLARE_SOA_COLUMN(Source, source, int);   //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,         //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //! absolute p
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} //namespace constituentssub
DECLARE_SOA_TABLE(JetConstituentsSub, "AOD", "CONSTITUENTSSUB", //!
                  constituentssub::JetId,
                  constituentssub::Pt,
                  constituentssub::Eta,
                  constituentssub::Phi,
                  constituentssub::Energy,
                  constituentssub::Mass,
                  constituentssub::Source,
                  constituentssub::Px<constituentssub::Pt, constituentssub::Phi>,
                  constituentssub::Py<constituentssub::Pt, constituentssub::Phi>,
                  constituentssub::Pz<constituentssub::Pt, constituentssub::Eta>,
                  constituentssub::P<constituentssub::Pt, constituentssub::Eta>);
using JetConstituentSub = JetConstituentsSub::iterator;

} // namespace o2::aod

#endif
