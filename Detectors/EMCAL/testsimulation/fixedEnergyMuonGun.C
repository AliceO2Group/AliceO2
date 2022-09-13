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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include "FairBoxGenerator.h"
#endif

FairGenerator* fixedEnergyMuonGun(double energy)
{
  std::cout << "Single muon generator shooting at EMCAL with Energy " << energy << "GeV/c" << std::endl;
  auto elecgen = new FairBoxGenerator(13, 1);
  elecgen->SetEtaRange(-0.67, 0.67);
  elecgen->SetPhiRange(90, 340);
  elecgen->SetPRange(energy, energy);
  return elecgen;
}