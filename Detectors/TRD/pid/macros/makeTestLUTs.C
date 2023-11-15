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
// ROOT header
#include <TFile.h>
#include <TGraph.h>

#include "TRDPID/LQND.h"

#include <vector>
#include <memory>
#endif

constexpr int dim = 1;

/// Generate very simple luts for testing
void makeTestLUTs()
{
  std::vector<float> p{1.0, 2.0, 3.0, 100.0};
  std::vector<TGraph> g;
  double x[4] = {0.0, 10, 70, 317};
  double y[4] = {0.0, 0.0, 1.0, 1.0};
  for (auto i = 0; i < dim * 2 * p.size(); ++i) {
    g.emplace_back(4, x, y);
  }

  o2::trd::detail::LUT<dim> luts(p, g);

  std::unique_ptr<TFile> outFile(TFile::Open("LQND_LUTS.root", "RECREATE"));
  outFile->WriteObject(&luts, "luts");
}
