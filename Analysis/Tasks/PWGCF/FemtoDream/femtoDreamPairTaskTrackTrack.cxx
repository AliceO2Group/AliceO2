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

/// \file femtoDreamPairTaskTrackTrack.cxx
/// \brief Tasks that reads the track tables used for the pairing and builds pairs of two tracks
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "include/FemtoDream/FemtoDerived.h"
#include "include/FemtoDream/FemtoDreamParticleHisto.h"
#include "include/FemtoDream/FemtoDreamPairCleaner.h"
#include "include/FemtoDream/FemtoDreamContainer.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::analysis::femtoDream;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct femtoDreamPairTaskTrackTrack {

  /// Particle selection part
  uint trackTypeSel = aod::femtodreamparticle::ParticleType::kTrack; // \todo at some point filters will be able to cope with enums

  /// Particle 1
  Configurable<int> ConfPDGCodePartOne{"ConfPDGCodePartOne", 2212, "Particle 1 - PDG code"};
  Configurable<aod::femtodreamparticle::cutContainerType> ConfCutPartOne{"ConfCutPartOne", 65534, "Particle 1 - Selection bit"};
  Configurable<float> ConfPtMinPartOne{"ConfPtMinPartOne", 0.f, "Particle 1 - min. pT selection (GeV/c)"};
  Configurable<float> ConfPtMaxPartOne{"ConfPtMaxPartOne", 999.f, "Particle 1 - max. pT selection (GeV/c)"};
  Configurable<float> ConfEtaMaxPartOne{"ConfEtaMaxPartOne", 999.f, "Particle 1 - max. eta selection"};
  Configurable<float> ConfDCAxyMaxPartOne{"ConfDCAxyMaxPartOne", 999.f, "Particle 1 - max. DCA_xy selection (cm)"};
  Configurable<float> ConfPIDThreshPartOne{"ConfPIDThreshPartOne", 0.75f, "Particle 1 - TPC / TPC+TOF PID momentum threshold"};
  Configurable<std::vector<int>> ConfTPCPIDPartOne{"ConfTPCPIDPartOne", std::vector<int>{7}, "Particle 1 - TPC PID bits"};        // we also need the possibility to specify whether the bit is true/false ->std>>vector<std::pair<int, int>>
  Configurable<std::vector<int>> ConfCombPIDPartOne{"ConfCombPIDPartOne", std::vector<int>{7}, "Particle 1 - Combined PID bits"}; // we also need the possibility to specify whether the bit is true/false ->std>>vector<std::pair<int, int>>

  /// Partition for particle 1
  Partition<aod::FemtoDreamParticles> partsOne = (aod::femtodreamparticle::partType == trackTypeSel) &&
                                                 (aod::femtodreamparticle::cut == ConfCutPartOne) &&
                                                 (aod::femtodreamparticle::pt > ConfPtMinPartOne) &&
                                                 (aod::femtodreamparticle::pt < ConfPtMaxPartOne) &&
                                                 (nabs(aod::femtodreamparticle::tempFitVar) < ConfDCAxyMaxPartOne) &&
                                                 (nabs(aod::femtodreamparticle::eta) < ConfEtaMaxPartOne);

  /// Histogramming for particle 1
  FemtoDreamParticleHisto<aod::femtodreamparticle::ParticleType::kTrack, 1> trackHistoPartOne;

  /// Particle 2
  Configurable<bool> ConfIsSame{"ConfIsSame", false, "Pairs of the same particle"};
  Configurable<int> ConfPDGCodePartTwo{"ConfPDGCodePartTwo", 2212, "Particle 2 - PDG code"};
  Configurable<aod::femtodreamparticle::cutContainerType> ConfCutPartTwo{"ConfCutPartTwo", 65534, "Particle 2 - Selection bit"};
  Configurable<float> ConfPtMinPartTwo{"ConfPtMinPartTwo", 0.5f, "Particle 2 - min. pT selection (GeV/c)"};
  Configurable<float> ConfPtMaxPartTwo{"ConfPtMaxPartTwo", 4.5f, "Particle 2 - max. pT selection (GeV/c)"};
  Configurable<float> ConfEtaMaxPartTwo{"ConfEtaMaxPartTwo", 999.f, "Particle 2 - max. eta selection"};
  Configurable<float> ConfDCAxyMaxPartTwo{"ConfDCAxyMaxPartTwo", 999.f, "Particle 2 - max. DCA_xy selection (cm)"};
  Configurable<float> ConfPIDThreshPartTwo{"ConfPIDThreshPartTwo", 0.4f, "Particle 2 - TPC / TPC+TOF PID momentum threshold"};
  Configurable<std::vector<int>> ConfTPCPIDPartTwo{"ConfTPCPIDPartTwo", std::vector<int>{6}, "Particle 2 - TPC PID bits"};        // we also need the possibility to specify whether the bit is true/false ->std>>vector<std::pair<int, int>>
  Configurable<std::vector<int>> ConfCombPIDPartTwo{"ConfCombPIDPartTwo", std::vector<int>{6}, "Particle 2 - Combined PID bits"}; // we also need the possibility to specify whether the bit is true/false ->std>>vector<std::pair<int, int>>

  /// Partition for particle 2
  Partition<aod::FemtoDreamParticles> partsTwo = (aod::femtodreamparticle::partType == trackTypeSel) &&
                                                 (aod::femtodreamparticle::cut == ConfCutPartTwo) &&
                                                 (aod::femtodreamparticle::pt > ConfPtMinPartTwo) &&
                                                 (aod::femtodreamparticle::pt < ConfPtMaxPartTwo) &&
                                                 (nabs(aod::femtodreamparticle::tempFitVar) < ConfDCAxyMaxPartTwo) &&
                                                 (nabs(aod::femtodreamparticle::eta) < ConfEtaMaxPartTwo);

  /// Histogramming for particle 2
  FemtoDreamParticleHisto<aod::femtodreamparticle::ParticleType::kTrack, 2> trackHistoPartTwo;

  /// The configurables need to be passed to an std::vector
  std::vector<int> vecTPCPIDPartOne, vecTPCPIDPartTwo, vecCombPIDPartOne, vecCombPIDPartTwo;

  /// Correlation part
  ConfigurableAxis CfgMultBins{"CfgMultBins", {VARIABLE_WIDTH, 0.0f, 20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 200.0f}, "Mixing bins - multiplicity"}; // \todo to be obtained from the hash task
  ConfigurableAxis CfgkstarBins{"CfgkstarBins", {5000, 0., 5.}, "binning kstar"};
  ConfigurableAxis CfgkTBins{"CfgkTBins", {70, 0., 7.}, "binning kT"};
  ConfigurableAxis CfgmTBins{"CfgmTBins", {70, 0., 7.}, "binning mT"};
  Configurable<int> ConfNEventsMix{"ConfNEventsMix", 5, "Number of events for mixing"};

  FemtoDreamContainer<femtoDreamContainer::EventType::same, femtoDreamContainer::Observable::kstar> sameEventCont;
  FemtoDreamContainer<femtoDreamContainer::EventType::mixed, femtoDreamContainer::Observable::kstar> mixedEventCont;
  FemtoDreamPairCleaner<aod::femtodreamparticle::ParticleType::kTrack, aod::femtodreamparticle::ParticleType::kTrack> pairCleaner;

  /// Histogram output
  HistogramRegistry qaRegistry{"TrackQA", {}, OutputObjHandlingPolicy::AnalysisObject};
  HistogramRegistry resultRegistry{"Correlations", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    trackHistoPartOne.init(&qaRegistry);
    if (!ConfIsSame) {
      trackHistoPartTwo.init(&qaRegistry);
    }

    sameEventCont.init(&resultRegistry, CfgkstarBins, CfgMultBins, CfgkTBins, CfgmTBins);
    sameEventCont.setPDGCodes(ConfPDGCodePartOne, ConfPDGCodePartTwo);

    mixedEventCont.init(&resultRegistry, CfgkstarBins, CfgMultBins, CfgkTBins, CfgmTBins);
    mixedEventCont.setPDGCodes(ConfPDGCodePartOne, ConfPDGCodePartTwo);

    pairCleaner.init(&qaRegistry);

    vecTPCPIDPartOne = ConfTPCPIDPartOne;
    vecTPCPIDPartTwo = ConfTPCPIDPartTwo;
    vecCombPIDPartOne = ConfCombPIDPartOne;
    vecCombPIDPartTwo = ConfCombPIDPartTwo;
  }

  /// function that checks whether the PID selection specified in the vectors is fulfilled
  /// \param pidcut Bit-wise container for the PID
  /// \param vec Vector with the different selections
  /// \return Whether the PID selection specified in the vectors is fulfilled
  bool isPIDSelected(aod::femtodreamparticle::cutContainerType const& pidcut, std::vector<int> const& vec)
  {
    bool pidSelection = true;
    for (auto it : vec) {
      //\todo we also need the possibility to specify whether the bit is true/false ->std>>vector<std::pair<int, int>>
      //if (!((pidcut >> it.first) & it.second)) {
      if (!((pidcut >> it) & 1)) {
        pidSelection = false;
      }
    }
    return pidSelection;
  };

  /// function that checks whether the PID selection specified in the vectors is fulfilled, depending on the momentum TPC or TPC+TOF PID is conducted
  /// \param pidcut Bit-wise container for the PID
  /// \param mom Momentum of the track
  /// \param pidThresh Momentum threshold that separates between TPC and TPC+TOF PID
  /// \param vecTPC Vector with the different selections for the TPC PID
  /// \param vecComb Vector with the different selections for the TPC+TOF PID
  /// \return Whether the PID selection is fulfilled
  bool isFullPIDSelected(aod::femtodreamparticle::cutContainerType const& pidCut, float const& mom, float const& pidThresh, std::vector<int> const& vecTPC, std::vector<int> const& vecComb)
  {
    bool pidSelection = true;
    if (mom < pidThresh) {
      /// TPC PID only
      pidSelection = isPIDSelected(pidCut, vecTPC);
    } else {
      /// TPC + TOF PID
      pidSelection = isPIDSelected(pidCut, vecComb);
    }
    return pidSelection;
  };

  /// This function processes the same event and takes care of all the histogramming
  /// \todo the trivial loops over the tracks should be factored out since they will be common to all combinations of T-T, T-V0, V0-V0, ...
  void processSameEvent(o2::aod::FemtoDreamCollision& col,
                        o2::aod::FemtoDreamParticles& parts)
  {

    const int multCol = col.multV0M();

    /// Histogramming same event
    for (auto& part : partsOne) {
      if (!isFullPIDSelected(part.pidcut(), part.p(), ConfPIDThreshPartOne, vecTPCPIDPartOne, vecCombPIDPartOne)) {
        continue;
      }
      trackHistoPartOne.fillQA(part);
    }

    if (!ConfIsSame) {
      for (auto& part : partsTwo) {
        if (!isFullPIDSelected(part.pidcut(), part.p(), ConfPIDThreshPartTwo, vecTPCPIDPartTwo, vecCombPIDPartTwo)) {
          continue;
        }
        trackHistoPartTwo.fillQA(part);
      }
    }

    /// Now build the combinations
    for (auto& [p1, p2] : combinations(partsOne, partsTwo)) {
      if (!isFullPIDSelected(p1.pidcut(), p1.p(), ConfPIDThreshPartOne, vecTPCPIDPartOne, vecCombPIDPartOne) || !isFullPIDSelected(p2.pidcut(), p2.p(), ConfPIDThreshPartTwo, vecTPCPIDPartTwo, vecCombPIDPartTwo)) {
        continue;
      }

      // track cleaning
      if (!pairCleaner.isCleanPair(p1, p2, parts)) {
        continue;
      }

      sameEventCont.setPair(p1, p2, multCol);
    }
  }

  /// This function processes the mixed event
  /// \todo the trivial loops over the collisions and tracks should be factored out since they will be common to all combinations of T-T, T-V0, V0-V0, ...
  void processMixedEvent(o2::aod::FemtoDreamCollisions& cols,
                         o2::aod::Hashes& hashes,
                         o2::aod::FemtoDreamParticles& parts)
  {
    cols.bindExternalIndices(&parts);
    auto particlesTuple = std::make_tuple(parts);
    AnalysisDataProcessorBuilder::GroupSlicer slicer(cols, particlesTuple);

    for (auto& [collision1, collision2] : soa::selfCombinations("fBin", ConfNEventsMix, -1, soa::join(hashes, cols), soa::join(hashes, cols))) {
      auto it1 = slicer.begin();
      auto it2 = slicer.begin();
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision1.index()) {
          it1 = slice;
          break;
        }
      }
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision2.index()) {
          it2 = slice;
          break;
        }
      }

      auto particles1 = std::get<aod::FemtoDreamParticles>(it1.associatedTables());
      particles1.bindExternalIndices(&cols);
      auto particles2 = std::get<aod::FemtoDreamParticles>(it2.associatedTables());
      particles2.bindExternalIndices(&cols);

      partsOne.bindTable(particles1);
      partsTwo.bindTable(particles2);

      /// \todo before mixing we should check whether both collisions contain a pair of particles!
      /// could work like that, but only if PID is contained within the partitioning!
      // auto particlesEvent1 = std::get<aod::FemtoDreamParticles>(it1.associatedTables());
      // particlesEvent1.bindExternalIndices(&cols);
      // auto particlesEvent2 = std::get<aod::FemtoDreamParticles>(it2.associatedTables());
      // particlesEvent2.bindExternalIndices(&cols);
      /// for the x-check
      // partsOne.bindTable(particlesEvent2);
      // auto nPart1Evt2 = partsOne.size();
      // partsTwo.bindTable(particlesEvent1);
      // auto nPart2Evt1 = partsTwo.size();
      /// for actual event mixing
      // partsOne.bindTable(particlesEvent1);
      // partsTwo.bindTable(particlesEvent2);
      // if (partsOne.size() == 0 || nPart2Evt1 == 0 || nPart1Evt2 == 0 || partsTwo.size() == 0 ) continue;

      for (auto& [p1, p2] : combinations(partsOne, partsTwo)) {
        if (!isFullPIDSelected(p1.pidcut(), p1.p(), ConfPIDThreshPartOne, vecTPCPIDPartOne, vecCombPIDPartOne) || !isFullPIDSelected(p2.pidcut(), p2.p(), ConfPIDThreshPartTwo, vecTPCPIDPartTwo, vecCombPIDPartTwo)) {
          continue;
        }

        mixedEventCont.setPair(p1, p2, collision1.multV0M()); // < \todo dirty trick, the multiplicity will be of course within the bin width used for the hashes
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<femtoDreamPairTaskTrackTrack>(cfgc, framework::Processes{&femtoDreamPairTaskTrackTrack::processSameEvent, &femtoDreamPairTaskTrackTrack::processMixedEvent}),
  };

  return workflow;
}
