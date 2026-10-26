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

// Writes the Geant4 particle table (one line per particle with a PDG code) to a text file.

#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4Version.hh"
#include "CLHEP/Units/PhysicalConstants.h"

#include "TGeant4.h"
#include "TG4RunConfiguration.h"

#include "SimConfig/G4Params.h"
#include "O2TrivialMC/O2TrivialMCApplication.h"

int main(int argc, char** argv)
{
  if (argc != 3 || std::string(argv[1]) != "--output") {
    std::cerr << "Usage: " << argv[0] << " --output <file>\n";
    return 1;
  }

  std::ofstream os{argv[2]};
  if (!os.is_open()) {
    std::cerr << "Cannot open " << argv[2] << "\n";
    return 1;
  }

  // We use our O2 VMC setup to make sure we are in line with our physics definition.
  new o2::mc::O2TrivialMCApplication();
  auto& physicsSetup = ::o2::conf::G4Params::Instance().getPhysicsConfigString();
  auto runConfiguration = new TG4RunConfiguration("geomRoot", physicsSetup);
  auto vmc = new TGeant4("TGeant4", "The Geant4 Monte Carlo", runConfiguration);
  // needs to be initialised to have the particle table / physics entirely ready
  vmc->Init();

  os << "# Geant4 particle table, Geant4 version " << G4VERSION_NUMBER << "\n";
  os << "# physics setup: " << physicsSetup << "\n";
  os << "# pdg g4name ctau_mm shortlived hasDecayTable\n";

  auto table = G4ParticleTable::GetParticleTable();
  auto it = table->GetIterator();
  it->reset();
  int nParticles = 0;
  while ((*it)()) {
    auto particle = it->value();
    int pdg = particle->GetPDGEncoding();
    if (pdg == 0) {
      continue;
    }
    double lifetime = particle->GetPDGLifeTime();
    // lifetime is in G4 internal units, times c_light gives mm
    double ctau = lifetime > 0. ? lifetime * CLHEP::c_light : 0.;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", ctau);
    os << pdg << " " << particle->GetParticleName() << " " << buf << " "
       << (particle->IsShortLived() ? 1 : 0) << " "
       << (particle->GetDecayTable() ? 1 : 0) << "\n";
    ++nParticles;
  }
  std::cout << "Wrote " << nParticles << " particles to " << argv[2] << "\n";
  return 0;
}
