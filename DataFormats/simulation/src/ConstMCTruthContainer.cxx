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

#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include <TFile.h>
#include <TTree.h>
#include <iostream>

using namespace o2::dataformats;

ConstMCTruthContainer<o2::MCCompLabel>* MCLabelIOHelper::loadFromTTree(TTree* tree, std::string const& brname, int entry)
{
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  o2::dataformats::IOMCTruthContainerView* labelROOTbuffer = nullptr;
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>* constlabels = new ConstMCTruthContainer<o2::MCCompLabel>();

  auto branch = tree->GetBranch(brname.c_str());
  if (!branch) {
    return nullptr;
  }
  auto labelClass = branch->GetClassName();
  bool oldlabelformat = false;

  // we check how the labels are persisted and react accordingly
  if (TString(labelClass).Contains("IOMCTruthContainer")) {
    branch->SetAddress(&labelROOTbuffer);
    branch->GetEntry(entry);
    if (labelROOTbuffer) {
      labelROOTbuffer->copyandflatten(*constlabels);
      delete labelROOTbuffer;
    }
  } else {
    // case when we directly streamed MCTruthContainer
    // TODO: support more cases such as ConstMCTruthContainer etc
    std::string serializedtype("o2::dataformats::MCTruthContainer<o2::MCCompLabel>");
    if (!TString(labelClass).EqualTo(serializedtype.c_str())) {
      std::cerr << "Error: expected serialized type " << serializedtype << " but found " << labelClass;
      return nullptr;
    }
    branch->SetAddress(&labels);
    branch->GetEntry(entry);
    if (labels) {
      labels->flatten_to(*constlabels);
      delete labels;
    }
  }
  return constlabels;
}
