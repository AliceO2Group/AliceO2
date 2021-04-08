// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// A utility for the purpose to produce a global merged TTree
// from multiple TTree (containing a subset of branches).
// A typical example is TPC clusterization/digitization: Clusters per TPC
// sector may sit in different files and we want to produce an aggregate TTree
// for further processing. The utility offers options to use TFriends or to make
// a deep copy.

#include <TTree.h>
#include <TFile.h>
#include <ROOT/RDataFrame.hxx>
#include <boost/program_options.hpp>
#include <set>
#include <vector>
#include <iostream>

struct Options {
  std::vector<std::string> infilenames;
  std::string treename;
  std::string outfilename;
  bool asfriend = false;
};

bool parseOptions(int argc, char* argv[], Options& optvalues)
{
  namespace bpo = boost::program_options;
  bpo::options_description options(
    "A tool to create a single TTree from a list of TTrees (each in its own file).\nMerging is "
    "done vertically - over branches - instead over entries (like in a TChain).\nIt corresponds to the TFriend mechanism but makes a deep copy\n"
    "(unless the friend is asked).\n\n"
    "Allowed options");

  options.add_options()(
    "infiles,i", bpo::value<std::vector<std::string>>(&optvalues.infilenames)->multitoken(), "All input files to be merged")(
    "treename,t", bpo::value<std::string>(&optvalues.treename), "Name of tree (assumed same in all files).")(
    "outfile,o", bpo::value<std::string>(&optvalues.outfilename)->default_value(""), "Outfile to be created with merged tree.")(
    "asfriend", "If merging is done using the friend mechanism.");
  options.add_options()("help,h", "Produce help message.");

  bpo::variables_map vm;
  try {
    bpo::store(bpo::command_line_parser(argc, argv).options(options).run(), vm);
    bpo::notify(vm);

    // help
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }
    if (vm.count("asfriend")) {
      optvalues.asfriend = true;
    }

  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing options; Available options:\n";
    std::cerr << options << std::endl;
    return false;
  }
  return true;
}

// Checks if all given files have a TTree of this name
// and if all entries are the same
// TODO: add more checks such as for non-overlapping branch names etc.
bool checkFiles(std::vector<std::string> const& filenames, std::string const& treename)
{
  bool ok = true;
  int entries = -1;
  for (auto& f : filenames) {
    TFile _tmpfile(f.c_str(), "OPEN");
    auto tree = (TTree*)_tmpfile.Get(treename.c_str());
    if (tree == nullptr) {
      ok = false;
      std::cerr << "File " << f << " doesn't have a tree of name " << treename;
    } else {
      if (entries == -1) {
        entries = tree->GetEntries();
      } else {
        if (entries != tree->GetEntries()) {
          std::cerr << "Trees have inconsistent number of entries ";
          ok = false;
        }
      }
    }
  }
  return ok;
}

void merge(Options const& options)
{
  if (options.asfriend) {
    // open the output file
    auto newfile = TFile::Open(options.outfilename.c_str(), "RECREATE");
    auto newtree = new TTree(options.treename.c_str(), "");
    // add remaining stuff as friend
    for (int i = 0; i < options.infilenames.size(); ++i) {
      newtree->AddFriend(options.treename.c_str(), options.infilenames[i].c_str());
    }
    newfile->Write();
    newfile->Close();

    // P. Canal suggests that this can be done in the following way to fix the branch names
    // in the merged file and to keep only the final file:
    //auto mainfile = TFile::Open(firsttreefilename, "UPDATE");
    //auto friendfile = TFile::Open(secondtreefilename, "READ");
    //auto friendtree = ffriendfile>Get<Tree>(secondtreename);
    //mainfile->cd();
    //auto friendcopy = friendtree->CloneTree(-1, "fast");
    //auto maintree = mainfile->Get<TTree>(firsttreename);
    //maintree->AddFriend(friendcopy);
    //mainfile->Write();
  } else {
    // NOTE: This is functional but potentially slow solution.
    // We should adapt this function as soon as more performant
    // ways are known.
    // See also: https://root-forum.cern.ch/t/make-a-new-ttree-from-a-deep-vertical-union-of-existing-ttrees/44250

    // open the first Tree
    TFile _tmpfile(options.infilenames[0].c_str(), "OPEN");
    auto t1 = (TTree*)_tmpfile.Get(options.treename.c_str());

    // add remaining stuff as friend
    for (int i = 1; i < options.infilenames.size(); ++i) {
      t1->AddFriend(options.treename.c_str(), options.infilenames[i].c_str());
    }
    ROOT::RDataFrame df(*t1);
    df.Snapshot(options.treename, options.outfilename);
  }
}

int main(int argc, char* argv[])
{
  Options optvalues;
  if (!parseOptions(argc, argv, optvalues)) {
    return 0;
  }

  auto ok = checkFiles(optvalues.infilenames, optvalues.treename);
  if (!ok) {
    return 1;
  }

  // merge files
  merge(optvalues);

  return 0;
}
