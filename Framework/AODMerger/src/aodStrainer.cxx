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

#include <map>
#include <list>
#include <fstream>
#include <getopt.h>
#include <sstream>
#include <random>

#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TObjString.h"
#include <TGrid.h>
#include <TMap.h>
#include <TLeaf.h>

#include "aodMerger.h"
#include <cinttypes>

// AOD strainer with correct index rewriting: a strainer only for the table of interest
int main(int argc, char* argv[])
{
  std::string inputAO2D("AO2D.root");
  std::string outputFileName{"AO2D_strained.root"};
  std::string tables{"O2bc,O2calotrigger,O2collision,O2fdd,O2ft0,O2fv0a"};
  double downsampling = 1.0;
  int verbosity = 2;
  int exitCode = 0; // 0: success, >0: failure

  std::random_device rd;  // Seed generator
  std::mt19937 gen(rd()); // Mersenne Twister generator
  std::uniform_real_distribution<> dis(0.0, 1.0);

  int option_index = 0;
  static struct option long_options[] = {
    {"input", required_argument, nullptr, 0},
    {"output", required_argument, nullptr, 1},
    {"verbosity", required_argument, nullptr, 2},
    {"tables", required_argument, nullptr, 3},
    {"downsampling", required_argument, nullptr, 4},
    {"help", no_argument, nullptr, 5},
    {nullptr, 0, nullptr, 0}};

  while (true) {
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1) {
      break;
    } else if (c == 0) {
      inputAO2D = optarg;
    } else if (c == 1) {
      outputFileName = optarg;
    } else if (c == 2) {
      verbosity = atoi(optarg);
    } else if (c == 3) {
      tables = optarg;
    } else if (c == 4) {
      downsampling = atof(optarg);
    } else if (c == 5) {
      printf("AO2D strainer tool. Options: \n");
      printf("  --input <%s>      Contains path to files to be merged. Default: %s\n", inputAO2D.c_str(), inputAO2D.c_str());
      printf("  --output <%s>   Target output ROOT file. Default: %s\n", outputFileName.c_str(), outputFileName.c_str());
      printf("  --verbosity <flag>           Verbosity of output (default: %d).\n", verbosity);
      printf("  --tables <list of tables>    Comma separated list of tables (default: %s).\n", tables.c_str());
      printf("  --downsampling <downsample>  Fraction of DF to be kept (default: %f)\n", downsampling);
      return -1;
    } else {
      return -2;
    }
  }

  printf("AOD strainer started with:\n");
  printf("  Input file: %s\n", inputAO2D.c_str());
  printf("  Output file name: %s\n", outputFileName.c_str());
  printf("  Tables to be kept: %s\n", tables.c_str());
  printf("  Downsampling: %f\n", downsampling);

  std::vector<std::string> listOfTables;
  std::stringstream ss(tables);
  std::string token;

  while (std::getline(ss, token, ',')) {
    listOfTables.push_back(token);
  }

  auto outputFile = TFile::Open(outputFileName.c_str(), "RECREATE", "", 505);
  TDirectory* outputDir = nullptr;
  TString line(inputAO2D.c_str());
  if (line.BeginsWith("alien://") && !gGrid && !TGrid::Connect("alien:")) {
    printf("Error: Could not connect to AliEn.\n");
    return -1;
  }
  printf("Processing input file: %s\n", line.Data());
  auto inputFile = TFile::Open(line);
  if (!inputFile) {
    printf("Error: Could not open input file %s.\n", line.Data());
    return -1;
  }

  TList* keyList = inputFile->GetListOfKeys();
  keyList->Sort();

  for (auto key1 : *keyList) {
    if (((TObjString*)key1)->GetString().EqualTo("metaData")) {
      auto metaDataCurrentFile = (TMap*)inputFile->Get("metaData");
      outputFile->cd();
      metaDataCurrentFile->Write("metaData", TObject::kSingleKey);
    }

    if (((TObjString*)key1)->GetString().EqualTo("parentFiles")) {
      auto parentFilesCurrentFile = (TMap*)inputFile->Get("parentFiles");
      outputFile->cd();
      parentFilesCurrentFile->Write("parentFiles", TObject::kSingleKey);
    }

    if (!((TObjString*)key1)->GetString().BeginsWith("DF_") || dis(gen) > downsampling) {
      continue;
    }

    auto dfName = ((TObjString*)key1)->GetString().Data();
    if (verbosity > 0) {
      printf("  Processing folder %s\n", dfName);
    }
    outputDir = outputFile->mkdir(dfName);
    auto folder = (TDirectoryFile*)inputFile->Get(dfName);
    auto treeList = folder->GetListOfKeys();

    treeList->Sort();

    // purging keys from duplicates
    for (auto i = 0; i < treeList->GetEntries(); ++i) {
      TKey* ki = (TKey*)treeList->At(i);
      for (int j = i + 1; j < treeList->GetEntries(); ++j) {
        TKey* kj = (TKey*)treeList->At(j);
        if (std::strcmp(ki->GetName(), kj->GetName()) == 0 && std::strcmp(ki->GetTitle(), kj->GetTitle()) == 0) {
          if (ki->GetCycle() < kj->GetCycle()) {
            printf("    *** FATAL *** we had ordered the keys, first cycle should be higher, please check");
            exitCode = 5;
          } else {
            // key is a duplicate, let's remove it
            treeList->Remove(kj);
            j--;
          }
        } else {
          // we changed key, since they are sorted, we won't have the same anymore
          break;
        }
      }
    }

    std::list<std::string> foundTrees;

    for (auto key2 : *treeList) {
      auto treeName = ((TObjString*)key2)->GetString().Data();
      bool found = (std::find(foundTrees.begin(), foundTrees.end(), treeName) != foundTrees.end());
      if (found == true) {
        printf("    ***WARNING*** Tree %s was already merged (even if we purged duplicated trees before, so this should not happen), skipping\n", treeName);
        continue;
      }
      bool foundTable = false;
      for (auto const& table : listOfTables) {
        if (table == removeVersionSuffix(treeName)) {
          foundTrees.push_back(treeName);
          foundTable = true;
          break;
        }
      }
      if (!foundTable) {
        if (verbosity > 2) {
          printf("    Skipping tree %s\n", treeName);
        }
        continue;
      }

      auto inputTree = (TTree*)inputFile->Get(Form("%s/%s", dfName, treeName));
      if (verbosity > 1) {
        printf("    Processing tree %s with %lld entries with total size %lld\n", treeName, inputTree->GetEntries(), inputTree->GetTotBytes());
      }

      outputDir->cd();
      auto outputTree = inputTree->CloneTree(-1, "fast");
      outputTree->Write();
    }
  }
  // in case of failure, remove the incomplete file
  if (exitCode != 0) {
    printf("Removing incomplete output file %s.\n", outputFile->GetName());
    gSystem->Unlink(outputFile->GetName());
  } else {
    outputFile->Close();
  }
  return exitCode;
}
