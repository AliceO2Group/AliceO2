// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TPCWorkflow
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <TFile.h>
#include <TTree.h>

using namespace o2;

BOOST_AUTO_TEST_CASE(TPCWorkflow_types)
{
  // check that all types used in the workflow can be written to tree
  // Background:
  // strangely enough, std::vector of MCLabelContainer can be serialized in a message
  // even though std::vector < o2::dataformats::MCTruthContainer < o2::MCCompLabel >>
  // has not been specified in the LinkDef file. But it can only e serialized to a
  // tree branch if it has been defined.
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  MCLabelContainer labels;
  std::vector<MCLabelContainer> containers;
  const char* filename = "testTPCWorkflowTypes.root";
  const char* treename = "testtree";
  std::unique_ptr<TFile> testFile(TFile::Open(filename, "RECREATE"));
  std::unique_ptr<TTree> testTree = std::make_unique<TTree>(treename, treename);

  auto* labelsobject = &labels;
  auto* labelsbranch = testTree->Branch("labels", &labelsobject);
  auto* containerobject = &containers;
  auto* containerbranch = testTree->Branch("containers", &containerobject);

  labelsbranch->Fill();
  containerbranch->Fill();

  testTree.release();
  testFile->Close();
}
