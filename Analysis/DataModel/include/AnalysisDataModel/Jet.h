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
#define JET_TABLE_DEF(_Name_,_Origin_,_Description_,_dummy_name_) \
    DECLARE_SOA_TABLE(_Name_, _Origin_, _Description_, \
                  o2::soa::Index<>, \
                  jet::CollisionId, \
                  jet::Pt, \
                  jet::Eta, \
                  jet::Phi, \
                  jet::Energy, \
                  jet::Mass, \
                  jet::Area, \
                  jet::Px<jet::Pt, jet::Phi>, \
                  jet::Py<jet::Pt, jet::Phi>, \
                  jet::Pz<jet::Pt, jet::Eta>, \
                  jet::P<jet::Pt, jet::Eta>, \
                  jet::Dummy##_dummy_name_<>); \
    DECLARE_SOA_EXTENDED_TABLE(Matched##_Name_, _Name_, _Description_"MATCH", \
                                jet::MatchedJetIndex); \

// Defines the jet constituent table
#define JET_CONSTITUENTS_TABLE_DEF(_jet_type_,_name_,_Description_,_track_type_) \
    namespace _name_##constituents { \
    DECLARE_SOA_INDEX_COLUMN(_jet_type_, jet);     \
    DECLARE_SOA_INDEX_COLUMN(_track_type_, track); \
    DECLARE_SOA_INDEX_COLUMN(EMCALCluster, cluster); \
    } \
    DECLARE_SOA_TABLE(_jet_type_##TrackConstituents, "AOD", _Description_"TRKCONSTS", \
                      _name_##constituents::_jet_type_##Id, \
                      _name_##constituents::_track_type_##Id); \
    DECLARE_SOA_TABLE(_jet_type_##ClusterConstituents, "AOD", _Description_"CLSCONSTS", \
                      _name_##constituents::_jet_type_##Id, \
                      _name_##constituents::EMCALClusterId);

// Defines the jet constituent sub table
#define JET_CONSTITUENTS_SUB_TABLE_DEF(_jet_type_,_name_,_Description_) \
    namespace _name_##constituentssub {  \
    DECLARE_SOA_INDEX_COLUMN(_jet_type_, jet); \
    DECLARE_SOA_COLUMN(Pt, pt, float); \
    DECLARE_SOA_COLUMN(Eta, eta, float); \
    DECLARE_SOA_COLUMN(Phi, phi, float); \
    DECLARE_SOA_COLUMN(Energy, energy, float); \
    DECLARE_SOA_COLUMN(Mass, mass, float); \
    DECLARE_SOA_COLUMN(Source, source, int); \
    DECLARE_SOA_DYNAMIC_COLUMN(Px, px, \
                            [](float pt, float phi) -> float { return pt * std::cos(phi); }); \
    DECLARE_SOA_DYNAMIC_COLUMN(Py, py, \
                            [](float pt, float phi) -> float { return pt * std::sin(phi); }); \
    DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, \
                            [](float pt, float eta) -> float { return pt * std::sinh(eta); }); \
    DECLARE_SOA_DYNAMIC_COLUMN(P, p, \
                            [](float pt, float eta) -> float { return pt * std::cosh(eta); }); \
    } \
    DECLARE_SOA_TABLE(_jet_type_##ConstituentsSub, "AOD", _Description_"CONSTSUB", \
                    _name_##constituentssub::_jet_type_##Id, \
                    _name_##constituentssub::Pt, \
                    _name_##constituentssub::Eta, \
                    _name_##constituentssub::Phi, \
                    _name_##constituentssub::Energy, \
                    _name_##constituentssub::Mass, \
                    _name_##constituentssub::Source, \
                    _name_##constituentssub::Px<_name_##constituentssub::Pt, _name_##constituentssub::Phi>, \
                    _name_##constituentssub::Py<_name_##constituentssub::Pt, _name_##constituentssub::Phi>, \
                    _name_##constituentssub::Pz<_name_##constituentssub::Pt, _name_##constituentssub::Eta>, \
                    _name_##constituentssub::P<_name_##constituentssub::Pt, _name_##constituentssub::Eta>);

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
// NOTE: These dummy values are required so that each table isn't considered to be identical.
//       They need a unique column, so we add one for each that does nothing and isn't accessed,
//       but makes it unique. Note that although the value is incremented, there is not reason
//       that this is required.
DECLARE_SOA_DYNAMIC_COLUMN(DummyData, dummyData,  //! Dummy for unique data table.
                            []() -> int { return 1; });
DECLARE_SOA_DYNAMIC_COLUMN(DummyMCParticleLevel, dummyMCParticleLevel,  //! Dummy for unique MC particle level table.
                            []() -> int { return 2; });
DECLARE_SOA_DYNAMIC_COLUMN(DummyMCDetectorLevel, dummyMCDetectorLevel,  //! Dummy for unique MC detector level table.
                            []() -> int { return 3; });
DECLARE_SOA_DYNAMIC_COLUMN(DummyHybridIntermediate, dummyHybridIntermediate,  //! Dummy for unique hybrid intermediate table.
                            []() -> int { return 4; });
} // namespace jet

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
JET_TABLE_DEF(Jets, "AOD", "JET", Data);
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
JET_TABLE_DEF(MCParticleLevelJets, "AOD", "JETMCPART", MCParticleLevel);
using MCParticleLevelJet = MCParticleLevelJets::iterator;
using MatchedMCParticleLevelJet = MatchedMCParticleLevelJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(MCParticleLevelJet, mcparticlelevel, "MCP", McParticle);
using MCParticleLevelJetTrackConstituent = MCParticleLevelJetTrackConstituents::iterator;
using MCParticleLevelJetClusterConstituent = MCParticleLevelJetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(MCParticleLevelJet, mcparticlelevel, "MCP");
using MCParticleLevelJetConstituentSub = MCParticleLevelJetConstituentsSub::iterator;

// MC Detector Level Jets
// NOTE: The same condition as describe for particle leve jets also applies here
//       to subtracted constituents.
JET_TABLE_DEF(MCDetectorLevelJets, "AOD", "JETMCDET", MCDetectorLevel);
using MCDetectorLevelJet = MCDetectorLevelJets::iterator;
using MatchedMCDetectorLevelJet = MatchedMCDetectorLevelJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(MCDetectorLevelJet, mcdetectorlevel, "MCD", Track);
using MCDetectorLevelJetTrackConstituent = MCDetectorLevelJetTrackConstituents::iterator;
using MCDetectorLevelJetClusterConstituent = MCDetectorLevelJetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(MCDetectorLevelJet, mcdetectorlevel, "MCD");
using MCDetectorLevelJetConstituentSub = MCDetectorLevelJetConstituentsSub::iterator;

// Hybrid intermediate
JET_TABLE_DEF(HybridIntermediateJets, "AOD", "JETHYBINT", HybridIntermediate);
using HybridIntermediateJet = HybridIntermediateJets::iterator;
using MatchedHybridIntermediateJet = MatchedHybridIntermediateJets::iterator;
JET_CONSTITUENTS_TABLE_DEF(HybridIntermediateJet, hybridintermediate, "HYBINT", Track);
using HybridIntermediateJetTrackConstituent = HybridIntermediateJetTrackConstituents::iterator;
using HybridIntermediateJetClusterConstituent = HybridIntermediateJetClusterConstituents::iterator;
JET_CONSTITUENTS_SUB_TABLE_DEF(HybridIntermediateJet, hybridintermediate, "HYBINT");
using HybridIntermediateJetConstituentSub = HybridIntermediateJetConstituentsSub::iterator;

/*

DECLARE_SOA_TABLE(MCParticleLevelJets, "AOD", "JETMCPART", //!
                  o2::soa::Index<>,
                  jet::CollisionId,
                  jet::Pt,
                  jet::Eta,
                  jet::Phi,
                  jet::Energy,
                  jet::Mass,
                  jet::Area,
                  jet::Px<jet::Pt, jet::Phi>,
                  jet::Py<jet::Pt, jet::Phi>,
                  jet::Pz<jet::Pt, jet::Eta>,
                  jet::P<jet::Pt, jet::Eta>,
                  jet::DummyData<>);

DECLARE_SOA_EXTENDED_TABLE(MCParticleLevelMatchedJets, MCParticleLevelJets, "JETMCPARTMATCH", //!
                            jet::MatchedJetIndex);

using MCParticleLevelJet = MCParticleLevelJets::iterator;
using MatchedMCParticleLevelJet = MCParticleLevelMatchedJets::iterator;

// TODO: absorb in jet table
// when list of references available
namespace mcparticlelevelconstituents
{
DECLARE_SOA_INDEX_COLUMN(MCParticleLevelJet, jet); //!
DECLARE_SOA_INDEX_COLUMN(Track, track); //!
DECLARE_SOA_INDEX_COLUMN(EMCALCluster, cluster); //!
} // namespace constituents

DECLARE_SOA_TABLE(MCParticleLevelJetTrackConstituents, "AOD", "MCPTRKCONSTS", //!
                  mcparticlelevelconstituents::MCParticleLevelJetId,
                  mcparticlelevelconstituents::TrackId);
DECLARE_SOA_TABLE(MCParticleLevelJetClusterConstituents, "AOD", "MCPCLUSCONSTS", //!
                  mcparticlelevelconstituents::MCParticleLevelJetId,
                  mcparticlelevelconstituents::EMCALClusterId);

using MCParticleLevelJetTrackConstituent = MCParticleLevelJetTrackConstituents::iterator;
using MCParticleLevelJetClusterConstituent = MCParticleLevelJetClusterConstituents::iterator;

// MC detector level
DECLARE_SOA_EXTENDED_TABLE(JetsMCDetectorLevel, Jets, "JETMCDET"); //!
DECLARE_SOA_EXTENDED_TABLE(JetsMatchedMCDetectorLevel, JetsMCDetectorLevel, "JETMCDETMATCH", //!
                            jet::MatchedJetIndex);
using JetMCDetectorLevel = JetsMCDetectorLevel::iterator;
using JetMCDetectorLevelMatched = JetsMatchedMCDetectorLevel::iterator;

// Hybrid intermediate level
DECLARE_SOA_EXTENDED_TABLE(JetsHybridIntermediate, Jets, "JETHYBINT"); //!
DECLARE_SOA_EXTENDED_TABLE(JetsMatchedHybridIntermediate, JetsHybridIntermediate, "JETHYBINTMATCH", //!
                            jet::MatchedJetIndex);
using JetHybridIntermediate = JetsHybridIntermediate::iterator;
using JetHybridIntermediateMatched = JetsMatchedHybridIntermediate::iterator;

// MC particle level. TODO: It doesn't match so nicely because it doesn't have a concept of clusters...
// MC detector level
DECLARE_SOA_EXTENDED_TABLE(JetMCDetectorLevelTrackConstituents, JetTrackConstituents, "JETMCDETTRKS"); //!
DECLARE_SOA_EXTENDED_TABLE(JetMCDetectorLevelClusterConstituents, JetClusterConstituents, "JETMCDETCLUS"); //!
using JetMCDetectorLevelTrackConstituent = JetMCDetectorLevelTrackConstituents::iterator;
using JetMCDetectorLevelClusterConstituent = JetMCDetectorLevelClusterConstituents::iterator;
// Hybrid intermediate
DECLARE_SOA_EXTENDED_TABLE(JetHybridIntermediateTrackConstituents, JetTrackConstituents, "JETHYBINTTRKS"); //!
DECLARE_SOA_EXTENDED_TABLE(JetHybridIntermediateClusterConstituents, JetClusterConstituents, "JETHYBINTCLUS"); //!
using JetHybridIntermediateTrackConstituent = JetHybridIntermediateTrackConstituents::iterator;
using JetHybridIntermediateClusterConstituent = JetHybridIntermediateClusterConstituents::iterator;

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

// MC is a bit off here because they don't have a concept of subtracted constituents. However,
// any empty table also doesn't cause any issues, and it will never be filled.
// MC particle level. TODO. It doesn't match so nicely because it doesn't have a concept of clusters...
//// MC detector level
//DECLARE_SOA_EXTENDED_TABLE(JetMCDetectorLevelConstituentsSub, JetConstituentsSub, "JETMCDETSTRKS"); //!
//using JetMCDetectorLevelConstituentSub = JetMCDetectorLevelConstituentsSub::iterator;
//// Hybrid intermediate
//DECLARE_SOA_EXTENDED_TABLE(JetHybridIntermediateConstituentsSub, JetConstituentsSub, "JETHYBINTSTRKS"); //!
//using JetHybridIntermediateConstituentSub = JetHybridIntermediateConstituentsSub::iterator;

*/

} // namespace o2::aod

#endif
