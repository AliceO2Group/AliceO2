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

// A utility for the purpose to produce a global merged TTree
// from multiple TTree (containing a subset of branches).
// A typical example is TPC clusterization/digitization: Clusters per TPC
// sector may sit in different files and we want to produce an aggregate TTree
// for further processing. The utility offers options to use TFriends or to make
// a deep copy.

#include <TTree.h>
#include <TFile.h>
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

// just to make a protected interface accessible
class MyTTreeHelper : public TTree
{
 public:
  TBranch* PublicBranchImp(const char* branchname, TClass* ptrClass, void* addobj, Int_t bufsize, Int_t splitlevel)
  {
    return BranchImp(branchname, ptrClass, addobj, bufsize, splitlevel);
  }
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

// a helper function taken from TTree.cxx
static char DataTypeToChar(EDataType datatype)
{
  // Return the leaflist 'char' for a given datatype.

  switch (datatype) {
    case kChar_t:
      return 'B';
    case kUChar_t:
      return 'b';
    case kBool_t:
      return 'O';
    case kShort_t:
      return 'S';
    case kUShort_t:
      return 's';
    case kCounter:
    case kInt_t:
      return 'I';
    case kUInt_t:
      return 'i';
    case kDouble_t:
      return 'D';
    case kDouble32_t:
      return 'd';
    case kFloat_t:
      return 'F';
    case kFloat16_t:
      return 'f';
    case kLong_t:
      return 'G';
    case kULong_t:
      return 'g';
    case kchar:
      return 0; // unsupported
    case kLong64_t:
      return 'L';
    case kULong64_t:
      return 'l';

    case kCharStar:
      return 'C';
    case kBits:
      return 0; //unsupported

    case kOther_t:
    case kNoType_t:
    default:
      return 0;
  }
  return 0;
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
    // a deep copy solution

    auto copyBranch = [](TTree* t, TBranch* br) -> bool {
      // Get data from original branch and copy to new
      // by using generic type/class information of old.
      // We are using some internals of the TTree implementation. (Luckily these
      // functions are not marked private ... so that we can still access them).
      TClass* clptr = nullptr;
      EDataType type;
      if (br->GetExpectedType(clptr, type) == 0) {
        char* data = nullptr;
        TBranch* newbr = nullptr;
        if (clptr != nullptr) {
          newbr = ((MyTTreeHelper*)t)->PublicBranchImp(br->GetName(), clptr, &data, 32000, br->GetSplitLevel());
        } else if (type != EDataType::kOther_t) {
          TString varname;
          varname.Form("%s/%c", br->GetName(), DataTypeToChar(type));
          newbr = t->Branch(br->GetName(), &data, varname.Data());
        } else {
          std::cerr << "Could not retrieve class/type information. Branch " << br->GetName() << "cannot be copied.\n";
          return false;
        }
        if (newbr) {
          br->SetAddress(&data);
          for (int e = 0; e < br->GetEntries(); ++e) {
            auto size = br->GetEntry(e);
            newbr->Fill();
          }
          br->ResetAddress();
          br->DropBaskets("all");
          return true;
          // TODO: data is leaking? (but deleting it here causes a crash)
        }
      } // end good
      return false;
    };

    TFile outfile(options.outfilename.c_str(), "RECREATE");
    auto outtree = new TTree(options.treename.c_str(), options.treename.c_str());
    // iterate over files and branches
    for (auto filename : options.infilenames) {
      TFile _tmp(filename.c_str(), "OPEN");
      auto t = (TTree*)_tmp.Get(options.treename.c_str());
      auto brlist = t->GetListOfBranches();
      for (int i = 0; i < brlist->GetEntries(); ++i) {
        auto br = (TBranch*)brlist->At(i);
        if (!copyBranch(outtree, br)) {
          std::cerr << "Error copying branch " << br->GetName() << "\n";
        }
      }
      outtree->SetEntries(t->GetEntries());
    }
    outfile.Write();
    outfile.Close();
  }

  // Note: There is/was also an elegant solution based on RDataFrames (snapshot) as discussed here:
  // https://root-forum.cern.ch/t/make-a-new-ttree-from-a-deep-vertical-union-of-existing-ttrees/44250
  // ... but this solution has problems since ROOT 6-24 since RDataFrame may change the internal type
  // of std::vector<> using a non-default allocator which may cause problem when reading data back.
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
