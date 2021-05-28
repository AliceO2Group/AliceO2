// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <map>
#include <fstream>
#include <getopt.h>

#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TDirectory.h"
#include "TObjString.h"
#include <TGrid.h>

// AOD merger with correct index rewriting
// No need to know the datamodel because the branch names follow a canonical standard (identified by fIndex)
int main(int argc, char* argv[])
{
  std::string inputCollection("input.txt");
  std::string outputFileName("AO2D.root");
  long maxDirSize = 100000000;

  int this_option_optind = optind ? optind : 1;
  int option_index = 0;
  static struct option long_options[] = {
    {"input", required_argument, nullptr, 0},
    {"output", required_argument, nullptr, 1},
    {"max-size", required_argument, nullptr, 2},
    {"help", no_argument, nullptr, 3},
    {nullptr, 0, nullptr, 0}};

  while (true) {
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1) {
      break;
    } else if (c == 0) {
      inputCollection = optarg;
    } else if (c == 1) {
      outputFileName = optarg;
    } else if (c == 2) {
      maxDirSize = atol(optarg);
    } else if (c == 3) {
      printf("AOD merging tool. Options: \n");
      printf("  --input <inputfile.txt>      Contains path to files to be merged. Default: %s\n", inputCollection.c_str());
      printf("  --output <outputfile.root>   Target output ROOT file. Default: %s\n", outputFileName.c_str());
      printf("  --max-size <size in Bytes>   Target directory size: %ld \n", maxDirSize);
      return -1;
    } else {
      return -2;
    }
  }

  printf("AOD merger started with:\n");
  printf("  Input file: %s\n", inputCollection.c_str());
  printf("  Ouput file name: %s\n", outputFileName.c_str());
  printf("  Maximal folder size (uncompressed): %ld\n", maxDirSize);

  std::map<std::string, TTree*> trees;
  std::map<std::string, int> offsets;

  auto outputFile = TFile::Open(outputFileName.c_str(), "RECREATE", "", 501);
  TDirectory* outputDir = nullptr;
  long currentDirSize = 0;

  std::ifstream in;
  in.open(inputCollection);
  TString line;
  bool connectedToAliEn = false;
  int mergedDFs = 0;
  while (in.good()) {
    in >> line;

    if (line.Length() == 0) {
      continue;
    }

    if (line.BeginsWith("alien:") && !connectedToAliEn) {
      printf("Connecting to AliEn...");
      TGrid::Connect("alien:");
      connectedToAliEn = true; // Only try once
    }

    printf("Processing input file: %s\n", line.Data());

    auto inputFile = TFile::Open(line);
    TList* keyList = inputFile->GetListOfKeys();
    keyList->Sort();

    for (auto key1 : *keyList) {
      if (!((TObjString*)key1)->GetString().BeginsWith("DF_")) {
        continue;
      }

      auto dfName = ((TObjString*)key1)->GetString().Data();

      printf("  Processing folder %s\n", dfName);
      ++mergedDFs;
      auto folder = (TDirectoryFile*)inputFile->Get(dfName);
      auto treeList = folder->GetListOfKeys();

      for (auto key2 : *treeList) {
        auto treeName = ((TObjString*)key2)->GetString().Data();

        printf("    Processing tree %s\n", treeName);
        auto inputTree = (TTree*)inputFile->Get(Form("%s/%s", dfName, treeName));

        if (trees.count(treeName) == 0) {
          // clone tree
          // NOTE Basket size etc. are copied in CloneTree()
          if (!outputDir) {
            outputDir = outputFile->mkdir(dfName);
            currentDirSize = 0;
            printf("Writing to output folder %s\n", dfName);
          }
          outputDir->cd();
          auto outputTree = inputTree->CloneTree(-1, "fast");
          outputTree->SetAutoFlush(0);
          trees[treeName] = outputTree;
          currentDirSize += inputTree->GetTotBytes();
        } else {
          // append tree
          auto outputTree = trees[treeName];

          outputTree->CopyAddresses(inputTree);

          // register index columns
          std::vector<std::pair<int*, int>> indexList;
          TObjArray* branches = inputTree->GetListOfBranches();
          for (int i = 0; i < branches->GetEntriesFast(); ++i) {
            TBranch* br = (TBranch*)branches->UncheckedAt(i);
            TString branchName(br->GetName());
            if (branchName.BeginsWith("fIndex")) {
              // Syntax: fIndex<Table>[_<Suffix>]
              branchName.Remove(0, 6);
              if (branchName.First("_") > 0) {
                branchName.Remove(branchName.First("_"));
              }
              branchName.Remove(branchName.Length() - 1); // remove s
              branchName.ToLower();
              branchName = "O2" + branchName;

              indexList.push_back({new int, offsets[branchName.Data()]});

              inputTree->SetBranchAddress(br->GetName(), indexList.back().first);
              outputTree->SetBranchAddress(br->GetName(), indexList.back().first);
            }
          }

          auto entries = inputTree->GetEntries();
          for (int i = 0; i < entries; i++) {
            inputTree->GetEntry(i);
            // shift index columns by offset
            for (const auto& idx : indexList) {
              // if negative, the index is unassigned. In this case, the different unassigned blocks have to get unique negative IDs
              if (*(idx.first) < 0) {
                *(idx.first) = -mergedDFs;
              } else {
                *(idx.first) += idx.second;
              }
            }
            int nbytes = outputTree->Fill();
            if (nbytes > 0) {
              currentDirSize += nbytes;
            }
          }

          for (const auto& idx : indexList) {
            delete idx.first;
          }

          delete inputTree;
        }
      }

      // update offsets
      for (auto const& tree : trees) {
        offsets[tree.first] = tree.second->GetEntries();
      }

      // check for not found tables
      for (auto const& offset : offsets) {
        if (trees.count(offset.first) == 0) {
          printf("ERROR: Index on %s but no tree found\n", offset.first.c_str());
        }
      }

      if (currentDirSize > maxDirSize) {
        printf("Maximum size reached: %ld. Closing folder.\n", currentDirSize);
        for (auto const& tree : trees) {
          //printf("Writing %s\n", tree.first.c_str());
          outputDir->cd();
          tree.second->Write();
          delete tree.second;
        }
        outputDir = nullptr;
        trees.clear();
        offsets.clear();
        mergedDFs = 0;
      }
    }
    inputFile->Close();
  }
  outputFile->Write();
  outputFile->Close();

  printf("AOD merger finished.\n");

  return 0;
}
