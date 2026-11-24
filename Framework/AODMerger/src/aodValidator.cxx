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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <getopt.h>

#include "TFile.h"
#include "TKey.h"
#include "TTree.h"
#include "TBranch.h"
#include "TDirectory.h"
#include <TGrid.h>

// Checks the structure of AO2D files, local or on the grid (alien://), for
// what an interrupted write or merge leaves behind: a file which was not
// closed (ROOT has to recover its keys), no DF folder, a DF folder with no
// tree, a branch with fewer entries than its tree, or a basket with no size
// or location. Only the metadata is read (one read per tree); --deep also
// reads every basket's key to check that it belongs to its branch (one small
// read per basket).
namespace
{
int validate(const char* url, bool deep, int maxReports)
{
  if (!std::strncmp(url, "alien:", 6) && !gGrid) {
    TGrid::Connect("alien:");
    if (!gGrid) {
      printf("ERROR: cannot connect to AliEn for %s\n", url);
      return -1;
    }
  }
  std::unique_ptr<TFile> file(TFile::Open(url));
  if (!file || file->IsZombie()) {
    printf("ERROR: cannot open %s\n", url);
    return -1;
  }
  const Long64_t size = file->GetSize();
  int problems = 0, dfs = 0, trees = 0, baskets = 0;
  auto report = [&](const char* df, const char* tree, std::string const& what) {
    if (problems++ < maxReports) {
      printf("BAD %s/%s: %s\n", df, tree, what.c_str());
    }
  };
  if (file->TestBit(TFile::kRecovered)) {
    report("", "", "the file was not closed: ROOT recovered its keys, the DF folders may be empty");
  }

  for (auto* key : TRangeDynCast<TKey>(file->GetListOfKeys())) {
    if (!key || std::strncmp(key->GetName(), "DF_", 3) != 0) {
      continue;
    }
    ++dfs;
    std::unique_ptr<TDirectory> dir((TDirectory*)key->ReadObj());
    if (!dir) {
      report(key->GetName(), "", "cannot read the folder");
      continue;
    }
    int treesInFolder = 0;
    for (auto* treeKey : TRangeDynCast<TKey>(dir->GetListOfKeys())) {
      if (!treeKey || std::strcmp(treeKey->GetClassName(), "TTree") != 0) {
        continue;
      }
      std::unique_ptr<TTree> tree((TTree*)treeKey->ReadObj());
      if (!tree) {
        report(key->GetName(), treeKey->GetName(), "cannot read the tree");
        continue;
      }
      ++trees;
      ++treesInFolder;
      TObjArray* branches = tree->GetListOfBranches();
      for (int i = 0; i < branches->GetEntriesFast(); ++i) {
        auto* br = (TBranch*)branches->UncheckedAt(i);
        if (br->GetEntries() != tree->GetEntries()) {
          report(key->GetName(), tree->GetName(), Form("branch %s has %lld entries, the tree %lld", br->GetName(), br->GetEntries(), tree->GetEntries()));
        }
        for (int b = 0; b < br->GetWriteBasket(); ++b) {
          ++baskets;
          Int_t bytes = br->GetBasketBytes()[b];
          Long64_t seek = br->GetBasketSeek(b);
          if (bytes <= 0 || seek <= 0 || seek + bytes > size) {
            report(key->GetName(), tree->GetName(), Form("branch %s basket %d: %d bytes at %lld", br->GetName(), b, bytes, seek));
            continue;
          }
          if (deep) {
            char buffer[256];
            if (file->ReadBuffer(buffer, seek, std::min<Long64_t>(sizeof(buffer), size - seek))) {
              report(key->GetName(), tree->GetName(), Form("branch %s basket %d: cannot read its key at %lld", br->GetName(), b, seek));
              continue;
            }
            TKey basketKey(file.get());
            char* cursor = buffer;
            basketKey.ReadKeyBuffer(cursor);
            if (std::strcmp(basketKey.GetName(), br->GetName()) != 0) {
              report(key->GetName(), tree->GetName(), Form("branch %s basket %d at %lld is a basket of %s", br->GetName(), b, seek, basketKey.GetName()));
            }
          }
        }
      }
    }
    if (treesInFolder == 0) {
      report(key->GetName(), "", "the folder has no tree");
    }
  }
  if (dfs == 0) {
    report("", "", "the file has no DF folder");
  }
  printf("%s: %s, %d DFs, %d trees, %d baskets checked%s\n", url,
         problems ? Form("%d PROBLEMS", problems) : "OK", dfs, trees, baskets, deep ? " (with keys)" : "");
  return problems;
}
} // namespace

int main(int argc, char* argv[])
{
  bool deep = false;
  int maxReports = 50;
  std::vector<std::string> files;

  static struct option longOptions[] = {
    {"input", required_argument, nullptr, 'i'},
    {"deep", no_argument, nullptr, 'd'},
    {"max-reports", required_argument, nullptr, 'm'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "", longOptions, nullptr)) != -1) {
    if (c == 'i') {
      std::ifstream in(optarg);
      if (!in) {
        printf("ERROR: cannot read %s\n", optarg);
        return 2;
      }
      for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line[0] != '#') {
          files.push_back(line);
        }
      }
    } else if (c == 'd') {
      deep = true;
    } else if (c == 'm') {
      maxReports = std::stoi(optarg);
    } else {
      printf(
        "AO2D validator: checks AO2D files for what an interrupted write or merge leaves behind:\n"
        "a file not closed, no DF folder, a DF folder with no tree, a branch with fewer entries\n"
        "than its tree, a basket with no size or location. Reads only metadata.\n\n"
        "Usage: %s [options] FILE...   (local paths or alien:// names)\n"
        "  --input LIST       also check the files listed in LIST, one per line\n"
        "  --deep             also read every basket's key (one small read per basket)\n"
        "  --max-reports N    print at most N problems per file (default 50)\n\n"
        "Exit code: 0 all files fine, 1 a file has problems, 2 a file could not be read.\n",
        argv[0]);
      return c == 'h' ? 0 : 2;
    }
  }
  for (int i = optind; i < argc; ++i) {
    files.push_back(argv[i]);
  }
  if (files.empty()) {
    printf("ERROR: no input file, see --help\n");
    return 2;
  }

  int bad = 0, unreadable = 0;
  for (auto const& f : files) {
    int problems = validate(f.c_str(), deep, maxReports);
    bad += problems > 0;
    unreadable += problems < 0;
  }
  if (files.size() > 1) {
    printf("%zu files: %d with problems, %d unreadable\n", files.size(), bad, unreadable);
  }
  return unreadable ? 2 : (bad ? 1 : 0);
}
