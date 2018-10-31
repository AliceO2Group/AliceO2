// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Ruben Shahoyan, ruben.shahoyan@cern.ch

#include "CommonUtils/RootChain.h"
#include <TString.h>
#include <FairLogger.h>
#include <fstream>

using namespace o2::utils;

std::unique_ptr<TChain> RootChain::load(const std::string trName, const std::string inpFile)
{
  // create chain from the single root file or list of files
  FairLogger* logger = FairLogger::GetLogger();
  std::unique_ptr<TChain> chain;
  if (trName.empty() || inpFile.empty()) {
    LOG(ERROR) << "Tree name or input file is not provided" << FairLogger::endl;
    return chain;
  }
  chain = std::make_unique<TChain>(trName.data());
  addFile(chain.get(), inpFile);
  LOG(INFO) << "Created chain " << chain->GetName() << " with " << chain->GetEntries()
            << " from " << inpFile << FairLogger::endl;
  return chain;
}

void RootChain::addFile(TChain* ch, const std::string inp)
{
  // add root file or files from the list extracted from the inp text file
  TString inpS = inp.data();
  if (inpS.EndsWith(".root")) {
    LOG(INFO) << "Adding " << inp << FairLogger::endl;
    ch->AddFile(inp.data());
  } else {
    std::ifstream inpF(inpS.Data());
    if (!inpF.good()) {
      LOG(ERROR) << "Failed to open input file " << inp << " as a text one" << FairLogger::endl;
      return;
    }
    //
    inpS.ReadLine(inpF);
    while (!inpS.IsNull()) {
      inpS = inpS.Strip(TString::kBoth, ' ');
      if (inpS.BeginsWith("//") || inpS.BeginsWith("#")) {
        inpS.ReadLine(inpF);
        continue;
      }
      inpS = inpS.Strip(TString::kBoth, ',');
      inpS = inpS.Strip(TString::kBoth, '"');
      addFile(ch, inpS.Data());
      inpS.ReadLine(inpF);
    }
  }
}
