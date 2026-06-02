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

#ifndef FluenceWeightCalculator_h
#define FluenceWeightCalculator_h
#include <vector>
#include <string>
#include <memory>
#include "TGraph.h"
//
// Static container class for damage weight funnctions in form of TGraphs
// The weights can be read from a csv file and stored in the graphs.
//
class FluenceWeightCalculator
{
 public:
  FluenceWeightCalculator() = delete;
  static void InitWeights(const std::string& filename);
  static void InitWeightsFromCSV(const std::string& filename);
  static double GetWeight(const int pdg, const double ekin);

 private:
  static std::unique_ptr<TGraph> neutronG;
  static std::unique_ptr<TGraph> protonG;
  static std::unique_ptr<TGraph> pionG;
};
#endif
