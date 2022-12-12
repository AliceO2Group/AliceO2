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

#include "TGeant4.h"
#include "TG4RunConfiguration.h"
#include "TG4G3Units.h"

#include "SimConfig/G4Params.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "O2TrivialMC/O2TrivialMCApplication.h"

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
  new o2::mc::O2TrivialMCApplication();
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
    if (line.empty()) {
      continue;
    }

    // Save the PDG...
    auto pdg = std::stoi(line);
    checkedIons.push_back(pdg);
    // ...and since we cannot look for isomeres, set the last digit to 0.
    // In that case we will get at least the approximate PDG properties if found in the G4 ion table.
    auto pdgToLookFor = pdg / 10 * 10;
    auto particle = g4IonTable->GetIon(pdgToLookFor);
    if (!particle) {
      stillUnknown.push_back(pdg);
      continue;
    }
    if (pdgDB->GetParticle(pdg) || pdgDB->GetParticle(pdgToLookFor)) {
      // here we can look if we have it already
      continue;
    }
    auto name = particle->GetParticleName();
    // add only the ground state for now. If needed, we can take care of isomers later on
    pdgDB->AddParticle(name.c_str(), name.c_str(), particle->GetPDGMass() / TG4G3Units::Energy(), particle->GetPDGStable(), particle->GetPDGWidth() / TG4G3Units::Energy(), particle->GetPDGCharge() * 3, "Ion", pdgToLookFor);
    addedIons.push_back(pdgToLookFor);
  }
  is.close();

  // remove potential duplicates that came from the input
  removeDuplicates(stillUnknown);
  removeDuplicates(checkedIons);
  removeDuplicates(addedIons);

  std::cout << "Ran over " << checkedIons.size() << " different PDGs"
            << "\nout of which " << addedIons.size() << " ground states were added."
            << "\nStill unkown (probably originating from event generator) " << stillUnknown.size() << "\n";
  for (auto& it : stillUnknown) {
    std::cout << "  " << it << "\n";
  }

  std::string pdgTableOut{"newPDGTable.dat"};
  pdgDB->WritePDGTable(pdgTableOut.c_str());
  std::cout << "New PDG table written to " << pdgTableOut << "\n";

  return 0;
}
