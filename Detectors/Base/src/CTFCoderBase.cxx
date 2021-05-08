// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CTFCoderBase.cxx
/// \brief Defintions for CTFCoderBase class (support of external dictionaries)
/// \author ruben.shahoyan@cern.ch

#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DetectorsBase/CTFCoderBase.h"
#include <filesystem>

using namespace o2::ctf;

template <typename T>
bool readFromTree(TTree& tree, const std::string brname, T& dest, int ev = 0)
{
  auto* br = tree.GetBranch(brname.c_str());
  if (br && br->GetEntries() > ev) {
    auto* ptr = &dest;
    br->SetAddress(&ptr);
    br->GetEntry(ev);
    br->ResetAddress();
    return true;
  }
  return false;
}

std::unique_ptr<TFile> CTFCoderBase::loadDictionaryTreeFile(const std::string& dictPath, bool mayFail)
{
  TDirectory* curd = gDirectory;
  std::unique_ptr<TFile> fileDict(!std::filesystem::exists(dictPath) ? nullptr : TFile::Open(dictPath.c_str()));
  if (!fileDict || fileDict->IsZombie()) {
    if (mayFail) {
      LOG(INFO) << "CTF dictionary file " << dictPath << " for detector " << mDet.getName() << " is absent, will use dictionaries stored in CTF";
      fileDict.reset();
      return std::move(fileDict);
    }
    LOG(ERROR) << "Failed to open CTF dictionary file " << dictPath << " for detector " << mDet.getName();
    throw std::runtime_error("Failed to open dictionary file");
  }
  auto tnm = std::string(o2::base::NameConf::CTFDICT);
  std::unique_ptr<TTree> tree((TTree*)fileDict->Get(tnm.c_str()));
  if (!tree) {
    fileDict.reset();
    LOG(ERROR) << "Did not find CTF dictionary tree " << tnm << " in " << dictPath;
    throw std::runtime_error("Did not fine CTF dictionary tree in the file");
  }
  CTFHeader ctfHeader;
  if (!readFromTree(*tree.get(), "CTFHeader", ctfHeader) || !ctfHeader.detectors[mDet]) {
    tree.reset();
    fileDict.reset();
    LOG(ERROR) << "Did not find CTF dictionary header or Detector " << mDet.getName() << " in it";
    if (!mayFail) {
      throw std::runtime_error("did not find CTFHeader with needed detector");
    }
  } else {
    LOG(INFO) << "Found CTF dictionary for " << mDet.getName() << " in " << dictPath;
  }
  return fileDict;
}
