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

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "G4ParticleTable.hh"
#include "G4IonTable.hh"

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"

#include "TVirtualMCApplication.h"
#include "TGeant4.h"
#include "TG4RunConfiguration.h"

#include "SimConfig/G4Params.h"
#include "SimulationDataFormat/O2DatabasePDG.h"

// That is an absolut minimal implementation of a TVirtualMCApplication
// Required to instantiate VMC (see below)
class MCAppDummy : public TVirtualMCApplication
{
 public:
  MCAppDummy() : TVirtualMCApplication("MCAppDummy", "MCAppDummy") {}
  ~MCAppDummy() override = default;
  MCAppDummy(MCAppDummy const& app) {}
  void ConstructGeometry() override
  {
    auto geoMgr = gGeoManager;
    // we need some dummies, any material and medium will do
    auto mat = new TGeoMaterial("vac", 0, 0, 0);
    auto med = new TGeoMedium("vac", 1, mat);
    auto vol = geoMgr->MakeBox("cave", med, 1, 1, 1);
    geoMgr->SetTopVolume(vol);
    geoMgr->CloseGeometry();
  }
  void InitGeometry() override {}
  void GeneratePrimaries() override {}
  void BeginEvent() override {}
  void BeginPrimary() override {}
  void PreTrack() override {}
  void Stepping() override {}
  void PostTrack() override {}
  void FinishPrimary() override {}
  void FinishEvent() override {}
  TVirtualMCApplication* CloneForWorker() const override
  {
    return new MCAppDummy(*this);
  }
};

void removeDuplicates(std::vector<int>& vec)
{
  // quick helper to erase duplicates from vectors
  std::sort(vec.begin(), vec.end());
  vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

int main(int argc, char** argv)
{
  // Takes a single text file as input. Expect simply single column with PDGs to be searched.
  if (argc != 2) {
    std::cerr << "Exactly one argument required containing the PDGs to look up.\n";
    return 1;
  }

  std::ifstream is{argv[1]};
  if (!is.is_open()) {
    std::cerr << "Cannot open " << argv[1] << "\n";
    return 1;
  }

  // We use our O2 VMC setup to make sure we are in line with our physics definition.
  // need a dummy VMC App
  new MCAppDummy();
  // setup G4 as we usually do
  auto& physicsSetup = ::o2::conf::G4Params::Instance().getPhysicsConfigString();
  auto runConfiguration = new TG4RunConfiguration("geomRoot", physicsSetup);
  auto vmc = new TGeant4("TGeant4", "The Geant4 Monte Carlo", runConfiguration);
  // needs to be initialised to have particle and ion tables / physics entirely ready
  vmc->Init();

  // G4 ion table
  auto g4IonTable = G4ParticleTable::GetParticleTable()->GetIonTable();
  // a new instance of TDatabasePDG to be filled
  auto pdgDB = o2::O2DatabasePDG::Instance();

  // Keep some basic stats for final output
  // Use these maps instead of vectors to avoid double counting - no need to to sort and .
  std::vector<int> stillUnknown;
  std::vector<int> checkedIons;
  std::vector<int> addedIons;

  std::string line;
  while (std::getline(is, line)) {

    // Save the PDG...
    auto pdg = std::stoi(line);
    checkedIons.push_back(pdg);
    // ...and since we cannot look for isomeres, set the last digit to 0.
    // In that case we will get at least the approximate PDG properties if found in the G4 ion table.
    line.back() = '0';
    auto pdgToLookFor = std::stoi(line);
    auto particle = g4IonTable->GetIon(pdgToLookFor);
    if (!particle) {
      stillUnknown.push_back(pdg);
      continue;
    }
    auto name = particle->GetParticleName();
    // but in the end we must add the actual - potentially isomeric - PDG
    auto ret = pdgDB->AddParticle(name.c_str(), name.c_str(), particle->GetPDGMass(), particle->GetPDGStable(), particle->GetPDGWidth(), particle->GetPDGCharge(), "Ion", pdg);
    if (ret) {
      addedIons.push_back(pdg);
    }
  }
  is.close();

  // remove potential duplicates that came from the input
  removeDuplicates(stillUnknown);
  removeDuplicates(checkedIons);

  std::cout << "Ran over " << checkedIons.size() << " different PDGs"
            << "\nout of which " << checkedIons.size() - addedIons.size() << " were already defined"
            << "\nwhile PDG properties of " << addedIons.size() << " were newly extracted."
            << "\nStill unkown (probably originating from event generator) " << stillUnknown.size() << "\n";
  for (auto& it : stillUnknown) {
    std::cout << "  " << it << "\n";
  }

  std::string pdgTableOut{"newPDGTable.dat"};
  pdgDB->WritePDGTable(pdgTableOut.c_str());
  std::cout << "New PDG table written to " << pdgTableOut << "\n";

  return 0;
}
