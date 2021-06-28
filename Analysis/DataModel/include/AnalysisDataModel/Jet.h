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

///
/// \brief Table definitions for jets
///
/// Since the JE framework requires a set of nearly identical tables, most the tables are
/// generated via macros. Usually this would be avoided, but maintaining a collection of
/// (nearly) identical tables was judged to be more the larger maintenance burden.
///
/// \author Jochen Klein
/// \author Nima Zardoshti
/// \author Raymond Ehlers

#ifndef O2_ANALYSIS_DATAMODEL_JET_H
#define O2_ANALYSIS_DATAMODEL_JET_H

#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/EMCALClusters.h"
#include <cmath>

// Defines the jet table definition
#define JET_TABLE_DEF(_collision_name_, _jet_type_, _name_, _description_) \
  namespace _name_##util                                                   \
  {                                                                        \
    DECLARE_SOA_DYNAMIC_COLUMN(Dummy##_jet_type_##s, dummy##_jet_type##s,  \
                               []() -> int { return 0; });                 \
  }                                                                        \
  DECLARE_SOA_TABLE(_jet_type_##s, "AOD", _description_,                   \
                    o2::soa::Index<>,                                      \
                    jet::_collision_name_##Id,                             \
                    jet::Pt,                                               \
                    jet::Eta,                                              \
                    jet::Phi,                                              \
                    jet::Energy,                                           \
                    jet::Mass,                                             \
                    jet::Area,                                             \
                    jet::R,                                                \
                    jet::Px<jet::Pt, jet::Phi>,                            \
                    jet::Py<jet::Pt, jet::Phi>,                            \
                    jet::Pz<jet::Pt, jet::Eta>,                            \
                    jet::P<jet::Pt, jet::Eta>,                             \
                    _name_##util::Dummy##_jet_type_##s<>);                 \
  namespace _name_##matching                                               \
  {                                                                        \
    DECLARE_SOA_INDEX_COLUMN(_jet_type_, jet);                             \
    DECLARE_SOA_COLUMN(MatchedJetIndex, matchedJetIndex, int);             \
  }                                                                        \
  DECLARE_SOA_TABLE(Matched##_jet_type_##s, "AOD", _description_ "MATCH",  \
                    _name_##matching::_jet_type_##Id,                      \
                    _name_##matching::MatchedJetIndex);

// Defines the jet constituent table
#define JET_CONSTITUENTS_TABLE_DEF(_jet_type_, _name_, _Description_, _track_type_)    \
  namespace _name_##constituents                                                       \
  {                                                                                    \
    DECLARE_SOA_INDEX_COLUMN(_jet_type_, jet);                                         \
    DECLARE_SOA_INDEX_COLUMN(_track_type_, track);                                     \
    DECLARE_SOA_INDEX_COLUMN(EMCALCluster, cluster);                                   \
  }                                                                                    \
  DECLARE_SOA_TABLE(_jet_type_##TrackConstituents, "AOD", _Description_ "TRKCONSTS",   \
                    _name_##constituents::_jet_type_##Id,                              \
                    _name_##constituents::_track_type_##Id);                           \
  DECLARE_SOA_TABLE(_jet_type_##ClusterConstituents, "AOD", _Description_ "CLSCONSTS", \
                    _name_##constituents::_jet_type_##Id,                              \
                    _name_##constituents::EMCALClusterId);

// Defines the jet constituent sub table
// NOTE: This relies on eth jet index column being defined in the constiteunts namespace.
//       Since these are always paired together, there's no point in redefining them.
#define JET_CONSTITUENTS_SUB_TABLE_DEF(_jet_type_, _name_, _Description_)           \
  DECLARE_SOA_TABLE(_jet_type_##ConstituentsSub, "AOD", _Description_ "CONSTSUB",   \
                    _name_##constituents::_jet_type_##Id,                           \
                    constituentssub::Pt,                                            \
                    constituentssub::Eta,                                           \
                    constituentssub::Phi,                                           \
                    constituentssub::Energy,                                        \
                    constituentssub::Mass,                                          \
                    constituentssub::Source,                                        \
                    constituentssub::Px<constituentssub::Pt, constituentssub::Phi>, \
                    constituentssub::Py<constituentssub::Pt, constituentssub::Phi>, \
                    constituentssub::Pz<constituentssub::Pt, constituentssub::Eta>, \
                    constituentssub::P<constituentssub::Pt, constituentssub::Eta>);

namespace o2::aod
{
namespace jet
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN(McCollision, mcCollision);
DECLARE_SOA_COLUMN(Pt, pt, float);              //!
DECLARE_SOA_COLUMN(Eta, eta, float);            //!
DECLARE_SOA_COLUMN(Phi, phi, float);            //!
DECLARE_SOA_COLUMN(Energy, energy, float);      //!
DECLARE_SOA_COLUMN(Mass, mass, float);          //!
DECLARE_SOA_COLUMN(Area, area, float);          //!
DECLARE_SOA_COLUMN(R, r, int);                  //!
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,              //!
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, //! absolute p
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace jet

// The standard constituents table is more simply defined fully via macros.

// Constituent sub
namespace constituentssub
{
// Jet index column will be added in the macro
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Energy, energy, float);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Source, source, int);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px,
                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py,
                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz,
                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p,
                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace constituentssub

// Data jets
// As an example, the expanded macros which are used to define the table is shown below.
// It represents that state of the table as of June 2021.
//
//DECLARE_SOA_TABLE(Jets, "AOD", "JET", //!
//                  o2::soa::Index<>,
//                  jet::CollisionId,
//                  jet::Pt,
//                  jet::Eta,
//                  jet::Phi,
//                  jet::Energy,
//                  jet::Mass,
//                  jet::Area,
//                  jet::Px<jet::Pt, jet::Phi>,
//                  jet::Py<jet::Pt, jet::Phi>,
//                  jet::Pz<jet::Pt, jet::Eta>,
//                  jet::P<jet::Pt, jet::Eta>);
//
//DECLARE_SOA_EXTENDED_TABLE(JetsMatched, Jets, "JETMATCHED", //!
//                            jet::MatchedJetIndex);
//
//using Jet = Jets::iterator;
//using JetMatched = JetsMatched::iterator;

// And for the constituents table:
// TODO: absorb in jet table when list of references available
//
//namespace constituents
//{
//DECLARE_SOA_INDEX_COLUMN(Jet, jet);     //!
//DECLARE_SOA_INDEX_COLUMN(Track, track); //!
//DECLARE_SOA_INDEX_COLUMN(EMCALCluster, cluster); //!
//} // namespace constituents
//DECLARE_SOA_TABLE(JetTrackConstituents, "AOD", "TRKCONSTITS", //!
//                  constituents::JetId,
//                  constituents::TrackId);
//DECLARE_SOA_TABLE(JetClusterConstituents, "AOD", "CLSCONSTITS", //!
//                  constituents::JetId,
//                  constituents::EMCALClusterId);
//using JetTrackConstituent = JetTrackConstituents::iterator;
//using JetClusterConstituent = JetClusterConstituents::iterator;

// And finally, the consitutent sub table
//
//namespace constituentssub
//{
//DECLARE_SOA_INDEX_COLUMN(Jet, jet);        //!
//DECLARE_SOA_COLUMN(Pt, pt, float);         //!
//DECLARE_SOA_COLUMN(Eta, eta, float);       //!
//DECLARE_SOA_COLUMN(Phi, phi, float);       //!
//DECLARE_SOA_COLUMN(Energy, energy, float); //!
//DECLARE_SOA_COLUMN(Mass, mass, float);     //!
//DECLARE_SOA_COLUMN(Source, source, int);   //!
//DECLARE_SOA_DYNAMIC_COLUMN(Px, px,         //!
//                           [](float pt, float phi) -> float { return pt * std::cos(phi); });
//DECLARE_SOA_DYNAMIC_COLUMN(Py, py, //!
//                           [](float pt, float phi) -> float { return pt * std::sin(phi); });
//DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, //!
//                           [](float pt, float eta) -> float { return pt * std::sinh(eta); });
//DECLARE_SOA_DYNAMIC_COLUMN(P, p, //! absolute p
//                           [](float pt, float eta) -> float { return pt * std::cosh(eta); });
//} //namespace constituentssub
//
//DECLARE_SOA_TABLE(JetConstituentsSub, "AOD", "CONSTITUENTSSUB", //!
//                  constituentssub::JetId,
//                  constituentssub::Pt,
//                  constituentssub::Eta,
//                  constituentssub::Phi,
//                  constituentssub::Energy,
//                  constituentssub::Mass,
//                  constituentssub::Source,
//                  constituentssub::Px<constituentssub::Pt, constituentssub::Phi>,
//                  constituentssub::Py<constituentssub::Pt, constituentssub::Phi>,
//                  constituentssub::Pz<constituentssub::Pt, constituentssub::Eta>,
//                  constituentssub::P<constituentssub::Pt, constituentssub::Eta>);
//using JetConstituentSub = JetConstituentsSub::iterator;

// Defining the tables via the macors.
// The using statements are kept separate for visbility.
// Data jets
JET_TABLE_DEF(Collision, Jet, jet, "JET");
using Jet = Jets::iterator;
using MatchedJet = MatchedJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(Jet, jet, "JET", Track);
using JetTrackConstituent = JetTrackConstituents::iterator;
using JetClusterConstituent = JetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(Jet, jet, "JET");
using JetConstituentSub = JetConstituentsSub::iterator;

// MC Particle Level Jets
// NOTE: Cluster constituents aren't really meaningful for particle level.
//       However, it's a convenient construction, as it allows everything else
//       to work as it would otherwise, and it won't be filled (because there
//       are no clusters and nothing that would be identified as clusters), so
//       it causes no harm. Perhaps better would be making this std::optional,
//       but for now, we keep it simple.
// NOTE: The same condition applies to subtracted constituents.
JET_TABLE_DEF(McCollision, MCParticleLevelJet, mcparticleleveljet, "JETMCPART");
using MCParticleLevelJet = MCParticleLevelJets::iterator;
using MatchedMCParticleLevelJet = MatchedMCParticleLevelJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(MCParticleLevelJet, mcparticleleveljet, "MCP", McParticle);
using MCParticleLevelJetTrackConstituent = MCParticleLevelJetTrackConstituents::iterator;
using MCParticleLevelJetClusterConstituent = MCParticleLevelJetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(MCParticleLevelJet, mcparticleleveljet, "MCP");
using MCParticleLevelJetConstituentSub = MCParticleLevelJetConstituentsSub::iterator;

// MC Detector Level Jets
// NOTE: The same condition as describe for particle leve jets also applies here
//       to subtracted constituents.
JET_TABLE_DEF(Collision, MCDetectorLevelJet, mcdetectorleveljet, "JETMCDET");
using MCDetectorLevelJet = MCDetectorLevelJets::iterator;
using MatchedMCDetectorLevelJet = MatchedMCDetectorLevelJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(MCDetectorLevelJet, mcdetectorleveljet, "MCD", Track);
using MCDetectorLevelJetTrackConstituent = MCDetectorLevelJetTrackConstituents::iterator;
using MCDetectorLevelJetClusterConstituent = MCDetectorLevelJetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(MCDetectorLevelJet, mcdetectorleveljet, "MCD");
using MCDetectorLevelJetConstituentSub = MCDetectorLevelJetConstituentsSub::iterator;

// Hybrid intermediate
JET_TABLE_DEF(Collision, HybridIntermediateJet, hybridintermediatejet, "JETHYBINT");
using HybridIntermediateJet = HybridIntermediateJets::iterator;
using MatchedHybridIntermediateJet = MatchedHybridIntermediateJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(HybridIntermediateJet, hybridintermediate, "HYBINT", Track);
using HybridIntermediateJetTrackConstituent = HybridIntermediateJetTrackConstituents::iterator;
using HybridIntermediateJetClusterConstituent = HybridIntermediateJetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(HybridIntermediateJet, hybridintermediate, "HYBINT");
using HybridIntermediateJetConstituentSub = HybridIntermediateJetConstituentsSub::iterator;

} // namespace o2::aod

#endif
