// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Sum the per-worker Geant4 scoring dumps of an o2-sim run into one file per mesh

#include "SimConfig/G4ScoringMerger.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[])
{
  const std::string directory = argc > 1 ? argv[1] : ".";
  const int expectedWorkers = argc > 2 ? std::atoi(argv[2]) : 0;
  const int merged = o2::conf::mergeG4ScoringDumps(directory, expectedWorkers);
  if (merged < 0) {
    return 1;
  }
  std::cout << "merged " << merged << " scoring mesh(es) in " << directory << "\n";
  return 0;
}
