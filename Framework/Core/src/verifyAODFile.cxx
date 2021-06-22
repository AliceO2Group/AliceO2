// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AnalysisDataModel.h"
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/Logger.h"
#include "Framework/ASoA.h"
#include <TFile.h>
#include <iostream>
#include <memory>

using namespace o2::framework;
using namespace o2::soa;

template <typename T>
void verifyTable(TFile* infile, const char* branchName)
{
  std::cout << "Table: " << o2::aod::MetadataTrait<T>::metadata::tableLabel() << std::endl;
  std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>(branchName, infile);
  TableBuilder builder;
  RootTableBuilderHelpers::convertASoA<T>(builder, *reader);
  auto table = builder.finalize();
  std::cout << table->schema()->ToString() << std::endl;
  std::cout << "---" << std::endl;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    LOG(ERROR) << "Bad number of arguments";
    return 1;
  }
  auto infile = std::make_unique<TFile>(argv[1]);
  if (infile.get() == nullptr || infile->IsOpen() == false) {
    LOG(ERROR) << "File not found: " << argv[1];
    return 1;
  }

  verifyTable<o2::aod::Collisions>(infile.get(), "O2collision");
  verifyTable<o2::aod::StoredTracks>(infile.get(), "O2track");
  verifyTable<o2::aod::StoredTracksCov>(infile.get(), "O2track");
  verifyTable<o2::aod::TracksExtra>(infile.get(), "O2track");
  verifyTable<o2::aod::Calos>(infile.get(), "O2calo");
  verifyTable<o2::aod::StoredFwdTracks>(infile.get(), "O2fwdtrack");
  return 0;
}
