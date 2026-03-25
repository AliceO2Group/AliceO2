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

/// @file   CMVContainer.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch

#include <stdexcept>
#include <fmt/format.h>

#include "TFile.h"

#include "TPCCalibration/CMVContainer.h"

namespace o2::tpc
{

std::unique_ptr<TTree> CMVPerTF::toTTree() const
{
  auto tree = std::make_unique<TTree>("ccdb_object", "ccdb_object");
  tree->SetAutoSave(0);
  tree->SetDirectory(nullptr);

  const CMVPerTF* ptr = this;
  tree->Branch("CMVPerTF", &ptr);
  tree->Fill();

  tree->ResetBranchAddresses();

  return tree;
}

void CMVPerTF::writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree) const
{
  TFile f(filename.c_str(), "RECREATE");
  if (f.IsZombie()) {
    throw std::runtime_error(fmt::format("CMVPerTF::writeToFile: cannot open '{}'", filename));
  }
  tree->Write();
  f.Close();
}

} // namespace o2::tpc