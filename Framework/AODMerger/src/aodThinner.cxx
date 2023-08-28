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

// AOD reduction tool
//   Designed for the 2022 pp data with specific selections:
//   - Remove all TPC only tracks
//   - Remove all V0s which refer to any removed track
//   - Remove all cascade which refer to any removed V0
//   - Remove all ambiguous track entries which point to a track with collision
//   - Adjust all indices
int main(int argc, char* argv[])
{
  std::string inputFileName("AO2D.root");
  std::string outputFileName("AO2D_thinned.root");
  int exitCode = 0; // 0: success, >0: failure

  int option_index = 1;
  static struct option long_options[] = {
    {"input", required_argument, nullptr, 0},
    {"output", required_argument, nullptr, 1},
    {"help", no_argument, nullptr, 2},
    {nullptr, 0, nullptr, 0}};

  while (true) {
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1) {
      break;
    } else if (c == 0) {
      inputFileName = optarg;
    } else if (c == 1) {
      outputFileName = optarg;
    } else if (c == 2) {
      printf("AO2D thinning tool. Options: \n");
      printf("  --input <inputfile.root>     Contains input file path to the file to be thinned. Default: %s\n", inputFileName.c_str());
      printf("  --output <outputfile.root>   Target output ROOT file. Default: %s\n", outputFileName.c_str());
      return -1;
    } else {
      return -2;
    }
  }

  printf("AOD reduction started with:\n");
  printf("  Input file: %s\n", inputFileName.c_str());
  printf("  Ouput file name: %s\n", outputFileName.c_str());

  auto outputFile = TFile::Open(outputFileName.c_str(), "RECREATE", "", 501);
  TDirectory* outputDir = nullptr;

  if (inputFileName.find("alien:") == 0) {
    printf("Connecting to AliEn...");
    TGrid::Connect("alien:");
  }

  auto inputFile = TFile::Open(inputFileName.c_str());
  if (!inputFile) {
    printf("Error: Could not open input file %s.\n", inputFileName.c_str());
    return 1;
  }

  TList* keyList = inputFile->GetListOfKeys();
  keyList->Sort();

  for (auto key1 : *keyList) {
    if (((TObjString*)key1)->GetString().EqualTo("metaData")) {
      auto metaData = (TMap*)inputFile->Get("metaData");
      outputFile->cd();
      metaData->Write("metaData", TObject::kSingleKey);
    }

    if (((TObjString*)key1)->GetString().EqualTo("parentFiles")) {
      auto parentFiles = (TMap*)inputFile->Get("parentFiles");
      outputFile->cd();
      parentFiles->Write("parentFiles", TObject::kSingleKey);
    }

    if (!((TObjString*)key1)->GetString().BeginsWith("DF_")) {
      continue;
    }

    auto dfName = ((TObjString*)key1)->GetString().Data();

    printf("  Processing folder %s\n", dfName);
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

    // Certain order needed in order to populate vectors of skipped entries
    auto v0Entry = (TObject*)treeList->FindObject("O2v0_001");
    treeList->Remove(v0Entry);
    treeList->AddFirst(v0Entry);

    // Prepare maps for track skimming
    auto trackExtraTree = (TTree*)inputFile->Get(Form("%s/%s", dfName, "O2trackextra")); // for example we can use this line to access the track table
    if (trackExtraTree == nullptr) {
      printf("O2trackextra table not found\n");
      exitCode = 6;
      break;
    }
    auto track_iu = (TTree*)inputFile->Get(Form("%s/%s", dfName, "O2track_iu"));
    if (track_iu == nullptr) {
      printf("O2track_iu table not found\n");
      exitCode = 7;
      break;
    }
    auto v0s = (TTree*)inputFile->Get(Form("%s/%s", dfName, "O2v0_001"));
    if (v0s == nullptr) {
      printf("O2v0_001 table not found\n");
      exitCode = 8;
      break;
    }

    std::vector<int> acceptedTracks(trackExtraTree->GetEntries(), -1);
    std::vector<bool> hasCollision(trackExtraTree->GetEntries(), false);
    std::vector<int> keepV0s(v0s->GetEntries(), -1);

    uint8_t tpcNClsFindable = 0;
    uint8_t ITSClusterMap = 0;
    uint8_t TRDPattern = 0;
    float_t TOFChi2 = 0;
    // char16_t fTPCNClsFindableMinusFound = 0;
    trackExtraTree->SetBranchAddress("fTPCNClsFindable", &tpcNClsFindable);
    trackExtraTree->SetBranchAddress("fITSClusterMap", &ITSClusterMap);
    trackExtraTree->SetBranchAddress("fTRDPattern", &TRDPattern);
    trackExtraTree->SetBranchAddress("fTOFChi2", &TOFChi2);
    // trackExtraTree->SetBranchAddress("fTPCNClsFindableMinusFound", &fTPCNClsFindableMinusFound);

    int fIndexCollisions = 0;
    track_iu->SetBranchAddress("fIndexCollisions", &fIndexCollisions);

    // loop over all tracks
    auto entries = trackExtraTree->GetEntries();
    int counter = 0;
    for (int i = 0; i < entries; i++) {
      trackExtraTree->GetEntry(i);
      track_iu->GetEntry(i);

      // Remove TPC only tracks
      if (tpcNClsFindable > 0. && ITSClusterMap == 0 && TRDPattern == 0 && TOFChi2 < -1.) {
        counter++;
      } else {
        acceptedTracks[i] = i - counter;
      }
      hasCollision[i] = (fIndexCollisions >= 0);
    }

    for (auto key2 : *treeList) {
      auto treeName = ((TObjString*)key2)->GetString().Data();

      auto inputTree = (TTree*)inputFile->Get(Form("%s/%s", dfName, treeName));
      printf("    Processing tree %s with %lld entries with total size %lld\n", treeName, inputTree->GetEntries(), inputTree->GetTotBytes());

      // Connect trees but do not copy entries (using the clone function)
      // NOTE Basket size etc. are copied in CloneTree()
      if (!outputDir) {
        outputDir = outputFile->mkdir(dfName);
        printf("Writing to output folder %s\n", dfName);
      }
      outputDir->cd();
      auto outputTree = inputTree->CloneTree(0);
      outputTree->SetAutoFlush(0);

      std::vector<int*> indexList;
      std::vector<char*> vlaPointers;
      std::vector<int*> indexPointers;
      TObjArray* branches = inputTree->GetListOfBranches();
      for (int i = 0; i < branches->GetEntriesFast(); ++i) {
        TBranch* br = (TBranch*)branches->UncheckedAt(i);
        TString branchName(br->GetName());
        TString tableName(getTableName(branchName, treeName));
        // register index of track index ONLY
        if (!tableName.EqualTo("O2track")) {
          continue;
        }
        // detect VLA
        if (((TLeaf*)br->GetListOfLeaves()->First())->GetLeafCount() != nullptr) {
          printf("  *** FATAL ***: VLA detection is not supported\n");
          exitCode = 9;
        } else if (branchName.BeginsWith("fIndexSlice")) {
          int* buffer = new int[2];
          memset(buffer, 0, 2 * sizeof(buffer[0]));
          vlaPointers.push_back(reinterpret_cast<char*>(buffer));
          inputTree->SetBranchAddress(br->GetName(), buffer);
          outputTree->SetBranchAddress(br->GetName(), buffer);

          indexList.push_back(buffer);
          indexList.push_back(buffer + 1);
        } else if (branchName.BeginsWith("fIndex") && !branchName.EndsWith("_size")) {
          int* buffer = new int;
          *buffer = 0;
          indexPointers.push_back(buffer);

          inputTree->SetBranchAddress(br->GetName(), buffer);
          outputTree->SetBranchAddress(br->GetName(), buffer);

          indexList.push_back(buffer);
        }
      }

      bool processingTracks = memcmp(treeName, "O2track", 7) == 0; // matches any of the track tables
      bool processingCascades = memcmp(treeName, "O2cascade", 9) == 0;
      bool processingV0s = memcmp(treeName, "O2v0", 4) == 0;
      bool processingAmbiguousTracks = memcmp(treeName, "O2ambiguoustrack", 16) == 0;

      auto indexV0s = -1;
      if (processingCascades) {
        inputTree->SetBranchAddress("fIndexV0s", &indexV0s);
        outputTree->SetBranchAddress("fIndexV0s", &indexV0s);
      }

      auto entries = inputTree->GetEntries();
      for (int i = 0; i < entries; i++) {
        inputTree->GetEntry(i);
        bool fillThisEntry = true;
        // Special case for Tracks, TracksExtra, TracksCov
        if (processingTracks) {
          if (acceptedTracks[i] < 0) {
            fillThisEntry = false;
          }
        } else {
          // Other table than Tracks* --> reassign indices to Tracks
          for (const auto& idx : indexList) {
            int oldTrackIndex = *idx;

            // if negative, the index is unassigned.
            if (oldTrackIndex >= 0) {
              if (acceptedTracks[oldTrackIndex] < 0) {
                fillThisEntry = false;
              } else {
                *idx = acceptedTracks[oldTrackIndex];
              }
            }
          }
        }

        // Reassign v0 index of cascades
        if (processingCascades) {
          if (keepV0s[indexV0s] < 0) {
            fillThisEntry = false;
          } else {
            indexV0s = keepV0s[indexV0s];
          }
        }

        // Keep only tracks which have no collision, see O2-3601
        if (processingAmbiguousTracks) {
          if (hasCollision[i]) {
            fillThisEntry = false;
          }
        }

        if (fillThisEntry) {
          outputTree->Fill();
          if (processingV0s) {
            keepV0s[i] = outputTree->GetEntries() - 1;
          }
        }
      }

      if (entries != outputTree->GetEntries()) {
        printf("      Reduced from %lld to %lld entries\n", entries, outputTree->GetEntries());
      }

      delete inputTree;

      for (auto& buffer : indexPointers) {
        delete buffer;
      }
      for (auto& buffer : vlaPointers) {
        delete[] buffer;
      }

      outputDir->cd();
      outputTree->Write();
      delete outputTree;
    }
    if (exitCode > 0) {
      break;
    }

    outputDir = nullptr;
  }
  inputFile->Close();

  outputFile->Write();
  outputFile->Close();

  // in case of failure, remove the incomplete file
  if (exitCode != 0) {
    printf("Removing incomplete output file %s.\n", outputFile->GetName());
    gSystem->Unlink(outputFile->GetName());
  }

  printf("End of AOD thinning.\n");

  return exitCode;
}
