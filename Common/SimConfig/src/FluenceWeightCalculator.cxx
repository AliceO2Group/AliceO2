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

#include "SimConfig/FluenceWeightCalculator.h"
#include <TFile.h>
#include <fstream>
#include <sstream>
#include <iostream>

std::unique_ptr<TGraph> FluenceWeightCalculator::neutronG;
std::unique_ptr<TGraph> FluenceWeightCalculator::protonG;
std::unique_ptr<TGraph> FluenceWeightCalculator::pionG;

double FluenceWeightCalculator::GetWeight(const int pdg, const double kineticEnergy)
{
  //
  // Obtain weight for given particle and kinetic energy
  if (!neutronG || !protonG || !pionG) {
    std::cerr << "FluenceWeightCalculator not initialized\n";
    return 0.;
  }
  switch (std::abs(pdg)) {
    case 2112: {
      return neutronG->Eval(kineticEnergy, nullptr, "S");
    }
    case 2212: {
      return ((kineticEnergy > 1e-3) ? protonG->Eval(kineticEnergy, nullptr, "S") : 0.);
    }
    case 211: {
      return ((kineticEnergy > 10.) ? pionG->Eval(kineticEnergy, nullptr, "S") : 0.);
    }
    default:
      return 0.0;
  }
}

void FluenceWeightCalculator::InitWeights(const std::string& filename)
{
  //
  // Read graphs from file
  TFile inFile(filename.c_str(), "READ");
  if (inFile.IsZombie()) {
    std::cerr << "Cannot open " << filename << "\n";
    return;
  }
  //
  TGraph* tmp = nullptr;
  inFile.GetObject("neutronDW", tmp);
  neutronG.reset(tmp ? static_cast<TGraph*>(tmp->Clone()) : nullptr);
  if (!neutronG) {
    std::cerr << "Missing graph neutronDW\n";
    return;
  }
  neutronG->SetBit(TGraph::kIsSortedX);
  inFile.GetObject("protonDW", tmp);
  protonG.reset(tmp ? static_cast<TGraph*>(tmp->Clone()) : nullptr);
  if (!protonG) {
    std::cerr << "Missing graph protonDW\n";
    return;
  }
  protonG->SetBit(TGraph::kIsSortedX);
  inFile.GetObject("pionDW", tmp);
  pionG.reset(tmp ? static_cast<TGraph*>(tmp->Clone()) : nullptr);
  if (!pionG) {
    std::cerr << "Missing graph pionDW\n";
    return;
  }
  pionG->SetBit(TGraph::kIsSortedX);
}

void FluenceWeightCalculator::InitWeightsFromCSV(const std::string& filename)
{
  //
  // read the NIEL weights from input file and store them as TGraphs
  neutronG = std::make_unique<TGraph>();
  neutronG->SetName("neutronDW");
  auto neuN = 0;
  protonG = std::make_unique<TGraph>();
  protonG->SetName("protonDW");
  auto proN = 0;
  pionG = std::make_unique<TGraph>();
  pionG->SetName("pionDW");
  auto pioN = 0;

  std::ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Error: cannot open file with damage weights.\n";
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    std::string particle, e_str, w_str;
    if (!std::getline(ss, particle, ',')) {
      continue;
    }
    if (!std::getline(ss, e_str, ',')) {
      continue;
    }
    if (!std::getline(ss, w_str, ',')) {
      continue;
    }
    auto e = std::stod(e_str);
    auto w = std::stod(w_str);
    auto pdg = std::stoi(particle);
    switch (pdg) {
      case 2112: {
        neutronG->SetPoint(neuN++, e, w);
        break;
      }
      case 2212: {
        protonG->SetPoint(proN++, e, w);
        break;
      }
      case 211: {
        pionG->SetPoint(pioN++, e, w);
        break;
      }
      default:;
    }
  }
  auto fout = new TFile("rd50_niel.root", "recreate");
  neutronG->Write();
  protonG->Write();
  pionG->Write();
  fout->Close();
}
