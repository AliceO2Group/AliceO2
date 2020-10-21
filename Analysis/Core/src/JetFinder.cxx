// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// jet finder task
//
// Author: Jochen Klein, Nima Zardoshti
#include "Analysis/JetFinder.h"
#include "Framework/AnalysisTask.h"

/// Sets the jet finding parameters
void JetFinder::setParams()
{

  if (!isReclustering) {
    jetEtaMin = etaMin + jetR; //in aliphysics this was (-etaMax + 0.95*jetR)
    jetEtaMax = etaMax - jetR;
  }

  //selGhosts =fastjet::SelectorRapRange(ghostEtaMin,ghostEtaMax) && fastjet::SelectorPhiRange(phiMin,phiMax);
  //ghostAreaSpec=fastjet::GhostedAreaSpec(selGhosts,ghostRepeatN,ghostArea,gridScatter,ktScatter,ghostktMean);
  ghostAreaSpec = fastjet::GhostedAreaSpec(ghostEtaMax, ghostRepeatN, ghostArea, gridScatter, ktScatter, ghostktMean); //the first argument is rapidity not pseudorapidity, to be checked
  jetDef = fastjet::JetDefinition(algorithm, jetR, recombScheme, strategy);
  areaDef = fastjet::AreaDefinition(areaType, ghostAreaSpec);
  selJets = fastjet::SelectorPtRange(jetPtMin, jetPtMax) && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax) && fastjet::SelectorPhiRange(jetPhiMin, jetPhiMax);
  jetDefBkg = fastjet::JetDefinition(algorithmBkg, jetBkgR, recombSchemeBkg, strategyBkg);
  areaDefBkg = fastjet::AreaDefinition(areaTypeBkg, ghostAreaSpec);
  selRho = fastjet::SelectorRapRange(bkgEtaMin, bkgEtaMax) && fastjet::SelectorPhiRange(bkgPhiMin, bkgPhiMax); //&& !fastjet::SelectorNHardest(2)    //here we have to put rap range, to be checked!
}

/// Sets the background subtraction estimater pointer
void JetFinder::setBkgE()
{
  if (bkgSubMode == BkgSubMode::rhoAreaSub || bkgSubMode == BkgSubMode::constSub) {
    bkgE = decltype(bkgE)(new fastjet::JetMedianBackgroundEstimator(selRho, jetDefBkg, areaDefBkg));
  } else {
    if (bkgSubMode != BkgSubMode::none) {
      LOGF(ERROR, "requested subtraction mode not implemented!");
    }
  }
}

/// Sets the background subtraction pointer
void JetFinder::setSub()
{
  //if rho < 1e-6 it is set to 1e-6 in AliPhysics
  if (bkgSubMode == BkgSubMode::rhoAreaSub) {
    sub = decltype(sub){new fastjet::Subtractor{bkgE.get()}};
  } else if (bkgSubMode == BkgSubMode::constSub) { //event or jetwise
    constituentSub = decltype(constituentSub){new fastjet::contrib::ConstituentSubtractor{bkgE.get()}};
    constituentSub->set_distance_type(fastjet::contrib::ConstituentSubtractor::deltaR);
    constituentSub->set_max_distance(constSubRMax);
    constituentSub->set_alpha(constSubAlpha);
    constituentSub->set_ghost_area(ghostArea);
    constituentSub->set_max_eta(bkgEtaMax);
    constituentSub->set_background_estimator(bkgE.get()); //what about rho_m
  } else {
    if (bkgSubMode != BkgSubMode::none) {
      LOGF(ERROR, "requested subtraction mode not implemented!");
    }
  }
}

/// Performs jet finding
/// \note the input particle and jet lists are passed by reference
/// \param inputParticles vector of input particles/tracks
/// \param jets veector of jets to be filled
/// \return ClusterSequenceArea object needed to access constituents
fastjet::ClusterSequenceArea JetFinder::findJets(std::vector<fastjet::PseudoJet>& inputParticles, std::vector<fastjet::PseudoJet>& jets) //ideally find a way of passing the cluster sequence as a reeference
{
  setParams();
  setBkgE();
  jets.clear();

  if (bkgE) {
    bkgE->set_particles(inputParticles);
    setSub();
  }
  if (constituentSub) {
    inputParticles = constituentSub->subtract_event(inputParticles);
  }
  fastjet::ClusterSequenceArea clusterSeq(inputParticles, jetDef, areaDef);
  jets = sub ? (*sub)(clusterSeq.inclusive_jets()) : clusterSeq.inclusive_jets();
  jets = selJets(jets);
  return clusterSeq;
}
