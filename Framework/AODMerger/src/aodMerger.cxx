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

// AOD merger with correct index rewriting
// No need to know the datamodel because the branch names follow a canonical standard (identified by fIndex)
int main(int argc, char* argv[])
{
  std::string inputCollection("input.txt");
  std::string outputFileName("AO2D.root");
  long maxDirSize = 100000000;
  bool skipNonExistingFiles = false;
  bool skipParentFilesList = false;
  int verbosity = 2;
  int exitCode = 0; // 0: success, >0: failure

  int option_index = 0;
  static struct option long_options[] = {
    {"input", required_argument, nullptr, 0},
    {"output", required_argument, nullptr, 1},
    {"max-size", required_argument, nullptr, 2},
    {"skip-non-existing-files", no_argument, nullptr, 3},
    {"skip-parent-files-list", no_argument, nullptr, 4},
    {"verbosity", required_argument, nullptr, 5},
    {"help", no_argument, nullptr, 6},
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
      skipNonExistingFiles = true;
    } else if (c == 4) {
      skipParentFilesList = true;
    } else if (c == 5) {
      verbosity = atoi(optarg);
    } else if (c == 6) {
      printf("AO2D merging tool. Options: \n");
      printf("  --input <inputfile.txt>      Contains path to files to be merged. Default: %s\n", inputCollection.c_str());
      printf("  --output <outputfile.root>   Target output ROOT file. Default: %s\n", outputFileName.c_str());
      printf("  --max-size <size in Bytes>   Target directory size. Default: %ld. Set to 0 if file is not self-contained.\n", maxDirSize);
      printf("  --skip-non-existing-files    Flag to allow skipping of non-existing files in the input list.\n");
      printf("  --skip-parent-files-list     Flag to allow skipping the merging of the parent files list.\n");
      printf("  --verbosity <flag>           Verbosity of output (default: %d).\n", verbosity);
      return -1;
    } else {
      return -2;
    }
  }

  printf("AOD merger started with:\n");
  printf("  Input file: %s\n", inputCollection.c_str());
  printf("  Output file name: %s\n", outputFileName.c_str());
  printf("  Maximal folder size (uncompressed): %ld\n", maxDirSize);
  if (skipNonExistingFiles) {
    printf("  WARNING: Skipping non-existing files.\n");
  }

  std::map<std::string, TTree*> trees;
  std::map<std::string, uint64_t> sizeCompressed;
  std::map<std::string, uint64_t> sizeUncompressed;
  std::map<std::string, int> offsets;
  std::map<std::string, int> unassignedIndexOffset;

  auto outputFile = TFile::Open(outputFileName.c_str(), "RECREATE", "", 501);
  TDirectory* outputDir = nullptr;
  long currentDirSize = 0;

  std::ifstream in;
  in.open(inputCollection);
  TString line;
  bool connectedToAliEn = false;
  TMap* metaData = nullptr;
  TMap* parentFiles = nullptr;
  int totalMergedDFs = 0;
  int mergedDFs = 0;
  while (in.good() && exitCode == 0) {
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
    if (!inputFile) {
      printf("Error: Could not open input file %s.\n", line.Data());
      if (skipNonExistingFiles) {
        continue;
      } else {
        printf("Aborting merge!\n");
        exitCode = 1;
        break;
      }
    }

    TList* keyList = inputFile->GetListOfKeys();
    keyList->Sort();

    for (auto key1 : *keyList) {
      if (((TObjString*)key1)->GetString().EqualTo("metaData")) {
        auto metaDataCurrentFile = (TMap*)inputFile->Get("metaData");
        if (metaData == nullptr) {
          metaData = metaDataCurrentFile;
          outputFile->cd();
          metaData->Write("metaData", TObject::kSingleKey);
        } else {
          for (auto metaDataPair : *metaData) {
            auto metaDataKey = ((TPair*)metaDataPair)->Key();
            if (metaDataCurrentFile->Contains(((TObjString*)metaDataKey)->GetString())) {
              auto value = (TObjString*)metaData->GetValue(((TObjString*)metaDataKey)->GetString());
              auto valueCurrentFile = (TObjString*)metaDataCurrentFile->GetValue(((TObjString*)metaDataKey)->GetString());
              if (!value->GetString().EqualTo(valueCurrentFile->GetString())) {
                printf("WARNING: Metadata differs between input files. Key %s : %s vs. %s\n", ((TObjString*)metaDataKey)->GetString().Data(),
                       value->GetString().Data(), valueCurrentFile->GetString().Data());
              }
            } else {
              printf("WARNING: Metadata differs between input files. Key %s is not present in current file\n", ((TObjString*)metaDataKey)->GetString().Data());
            }
          }
        }
      }

      if (((TObjString*)key1)->GetString().EqualTo("parentFiles") && !skipParentFilesList) {
        auto parentFilesCurrentFile = (TMap*)inputFile->Get("parentFiles");
        if (parentFiles == nullptr) {
          parentFiles = new TMap;
        }
        for (auto pair : *parentFilesCurrentFile) {
          parentFiles->Add(((TPair*)pair)->Key(), ((TPair*)pair)->Value());
        }
        delete parentFilesCurrentFile;
      }

      if (!((TObjString*)key1)->GetString().BeginsWith("DF_")) {
        continue;
      }

      auto dfName = ((TObjString*)key1)->GetString().Data();

      if (verbosity > 0) {
        printf("  Processing folder %s\n", dfName);
      }
      ++mergedDFs;
      ++totalMergedDFs;
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
        foundTrees.push_back(treeName);

        auto inputTree = (TTree*)inputFile->Get(Form("%s/%s", dfName, treeName));
        bool fastCopy = (inputTree->GetTotBytes() > 10000000); // Only do this for large enough trees to avoid that baskets are too small
        if (verbosity > 1) {
          printf("    Processing tree %s with %lld entries with total size %lld (fast copy: %d)\n", treeName, inputTree->GetEntries(), inputTree->GetTotBytes(), fastCopy);
        }

        bool alreadyCopied = false;
        if (trees.count(treeName) == 0) {
          if (mergedDFs > 1) {
            printf("    *** FATAL ***: The tree %s was not in the previous dataframe(s)\n", treeName);
            exitCode = 3;
          }

          // Connect trees but do not copy entries (using the clone function) unless fast copy is on
          // NOTE Basket size etc. are copied in CloneTree()
          if (!outputDir) {
            outputDir = outputFile->mkdir(dfName);
            currentDirSize = 0;
            if (verbosity > 0) {
              printf("Writing to output folder %s\n", dfName);
            }
          }
          outputDir->cd();
          auto outputTree = inputTree->CloneTree(-1, (fastCopy) ? "fast" : "");
          currentDirSize += inputTree->GetTotBytes(); // NOTE outputTree->GetTotBytes() is 0, so we use the inputTree here
          alreadyCopied = true;
          outputTree->SetAutoFlush(0);
          trees[treeName] = outputTree;
        } else {
          // adjust addresses tree
          trees[treeName]->CopyAddresses(inputTree);
        }

        auto outputTree = trees[treeName];
        // register index and connect VLA columns
        std::vector<std::pair<int*, int>> indexList;
        std::vector<char*> vlaPointers;
        std::vector<int*> indexPointers;
        TObjArray* branches = inputTree->GetListOfBranches();
        for (int i = 0; i < branches->GetEntriesFast(); ++i) {
          TBranch* br = (TBranch*)branches->UncheckedAt(i);
          TString branchName(br->GetName());

          // detect VLA
          if (((TLeaf*)br->GetListOfLeaves()->First())->GetLeafCount() != nullptr) {
            int maximum = ((TLeaf*)br->GetListOfLeaves()->First())->GetLeafCount()->GetMaximum();

            // get type
            static TClass* cls;
            EDataType type;
            br->GetExpectedType(cls, type);
            auto typeSize = TDataType::GetDataType(type)->Size();

            char* buffer = new char[maximum * typeSize];
            memset(buffer, 0, maximum * typeSize);
            vlaPointers.push_back(buffer);
            if (verbosity > 2) {
              printf("      Allocated VLA buffer of length %d with %d bytes each for branch name %s\n", maximum, typeSize, br->GetName());
            }
            inputTree->SetBranchAddress(br->GetName(), buffer);
            outputTree->SetBranchAddress(br->GetName(), buffer);

            if (branchName.BeginsWith("fIndexArray")) {
              for (int i = 0; i < maximum; i++) {
                indexList.push_back({reinterpret_cast<int*>(buffer + i * typeSize), offsets[getTableName(branchName, treeName)]});
              }
            }
          } else if (branchName.BeginsWith("fIndexSlice")) {
            int* buffer = new int[2];
            memset(buffer, 0, 2 * sizeof(buffer[0]));
            vlaPointers.push_back(reinterpret_cast<char*>(buffer));

            inputTree->SetBranchAddress(br->GetName(), buffer);
            outputTree->SetBranchAddress(br->GetName(), buffer);

            indexList.push_back({buffer, offsets[getTableName(branchName, treeName)]});
            indexList.push_back({buffer + 1, offsets[getTableName(branchName, treeName)]});
          } else if (branchName.BeginsWith("fIndex") && !branchName.EndsWith("_size")) {
            int* buffer = new int;
            *buffer = 0;
            indexPointers.push_back(buffer);

            inputTree->SetBranchAddress(br->GetName(), buffer);
            outputTree->SetBranchAddress(br->GetName(), buffer);

            indexList.push_back({buffer, offsets[getTableName(branchName, treeName)]});
          }
        }

        if (indexList.size() > 0) {
          auto entries = inputTree->GetEntries();
          int minIndexOffset = unassignedIndexOffset[treeName];
          auto newMinIndexOffset = minIndexOffset;
          for (int i = 0; i < entries; i++) {
            for (auto& index : indexList) {
              *(index.first) = 0; // Any positive number will do, in any case it will not be filled in the output. Otherwise the previous entry is used and manipulated in the following.
            }
            inputTree->GetEntry(i);
            // shift index columns by offset
            for (const auto& idx : indexList) {
              // if negative, the index is unassigned. In this case, the different unassigned blocks have to get unique negative IDs
              if (*(idx.first) < 0) {
                *(idx.first) += minIndexOffset;
                newMinIndexOffset = std::min(newMinIndexOffset, *(idx.first));
              } else {
                *(idx.first) += idx.second;
              }
            }
            if (!alreadyCopied) {
              int nbytes = outputTree->Fill();
              if (nbytes > 0) {
                currentDirSize += nbytes;
              }
            }
          }
          unassignedIndexOffset[treeName] = newMinIndexOffset;
        } else if (!alreadyCopied) {
          auto nbytes = outputTree->CopyEntries(inputTree, -1, (fastCopy) ? "fast" : "");
          if (nbytes > 0) {
            currentDirSize += nbytes;
          }
        }

        delete inputTree;

        for (auto& buffer : indexPointers) {
          delete buffer;
        }
        for (auto& buffer : vlaPointers) {
          delete[] buffer;
        }
      }
      if (exitCode > 0) {
        break;
      }

      // check if all trees were present
      if (mergedDFs > 1) {
        for (auto const& tree : trees) {
          bool found = (std::find(foundTrees.begin(), foundTrees.end(), tree.first) != foundTrees.end());
          if (found == false) {
            printf("  *** FATAL ***: The tree %s was not in the current dataframe\n", tree.first.c_str());
            exitCode = 4;
          }
        }
      }

      // set to -1 to identify not found tables
      for (auto& offset : offsets) {
        offset.second = -1;
      }

      // update offsets
      for (auto const& tree : trees) {
        offsets[removeVersionSuffix(tree.first.c_str())] = tree.second->GetEntries();
      }

      // check for not found tables
      for (auto& offset : offsets) {
        if (offset.second < 0) {
          if (maxDirSize > 0) {
            // if maxDirSize is 0 then we do not merge DFs and this error is not an error actually (e.g. for not self-contained derived data)
            printf("ERROR: Index on %s but no tree found\n", offset.first.c_str());
          }
          offset.second = 0;
        }
      }

      if (currentDirSize > maxDirSize) {
        if (verbosity > 0) {
          printf("Maximum size reached: %ld. Closing folder %s.\n", currentDirSize, dfName);
        }
        for (auto const& tree : trees) {
          // printf("Writing %s\n", tree.first.c_str());
          outputDir->cd();
          tree.second->Write();

          // stats
          sizeCompressed[tree.first] += tree.second->GetZipBytes();
          sizeUncompressed[tree.first] += tree.second->GetTotBytes();

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

  if (parentFiles) {
    outputFile->cd();
    parentFiles->Write("parentFiles", TObject::kSingleKey);
  }

  for (auto const& tree : trees) {
    outputDir->cd();
    tree.second->Write();

    // stats
    sizeCompressed[tree.first] += tree.second->GetZipBytes();
    sizeUncompressed[tree.first] += tree.second->GetTotBytes();

    delete tree.second;
  }

  outputFile->Write();
  outputFile->Close();

  if (totalMergedDFs == 0) {
    printf("ERROR: Did not merge a single DF. This does not seem right.\n");
    exitCode = 2;
  }

  // in case of failure, remove the incomplete file
  if (exitCode != 0) {
    printf("Removing incomplete output file %s.\n", outputFile->GetName());
    gSystem->Unlink(outputFile->GetName());
  } else {
    printf("AOD merger finished. Size overview follows:\n");

    uint64_t totalCompressed = 0;
    uint64_t totalUncompressed = 0;
    for (auto const& tree : sizeCompressed) {
      totalCompressed += tree.second;
      totalUncompressed += sizeUncompressed[tree.first];
    }
    if (totalCompressed > 0 && totalUncompressed > 0) {
      for (auto const& tree : sizeCompressed) {
        printf("  Tree %20s | Compressed: %12llu (%2.0f%%) | Uncompressed: %12llu (%2.0f%%)\n", tree.first.c_str(), tree.second, 100.0 * tree.second / totalCompressed, sizeUncompressed[tree.first], 100.0 * sizeUncompressed[tree.first] / totalUncompressed);
      }
    }
  }
  printf("\n");

  return exitCode;
}
