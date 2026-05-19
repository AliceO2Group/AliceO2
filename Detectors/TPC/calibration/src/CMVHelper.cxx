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

/// @file   CMVHelper.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Helper utilities for reading CMV ROOT files

#include "TPCCalibration/CMVHelper.h"

#include <iostream>

#include "TPCCalibration/CMVContainer.h"
#include "TParameter.h"

namespace o2::tpc
{

bool CMVFileHandle::open(const std::string& path)
{
  file = TFile::Open(path.c_str());
  if (!file || file->IsZombie()) {
    std::cerr << "CMVFileHandle: failed to open: " << path << "\n";
    return false;
  }
  file->GetObject("ccdb_object", tree);
  if (!tree) {
    std::cerr << "CMVFileHandle: TTree 'ccdb_object' not found in: " << path << "\n";
    close();
    return false;
  }

  // Extract firstTF / lastTF from UserInfo if stored by the aggregation workflow
  if (auto* ui = tree->GetUserInfo()) {
    if (auto* p = dynamic_cast<TParameter<long>*>(ui->FindObject("firstTF"))) {
      firstTFInTree = p->GetVal();
    }
    if (auto* p = dynamic_cast<TParameter<long>*>(ui->FindObject("lastTF"))) {
      lastTFInTree = p->GetVal();
    }
  }

  isCompressed = (tree->GetBranch("CMVPerTFCompressed") != nullptr);
  const bool isRaw = (tree->GetBranch("CMVPerTF") != nullptr);
  if (!isCompressed && !isRaw) {
    std::cerr << "CMVFileHandle: no recognised branch (CMVPerTFCompressed / CMVPerTF) in: "
              << path << "\n";
    close();
    return false;
  }

  if (isCompressed) {
    tree->SetBranchAddress("CMVPerTFCompressed", &tfCompressed);
    tfDecoded = new CMVPerTF();
  } else {
    tree->SetBranchAddress("CMVPerTF", &tfRaw);
  }
  return true;
}

const CMVPerTF* CMVFileHandle::getEntry(long long iEntry)
{
  tree->GetEntry(iEntry);
  if (isCompressed) {
    if (!tfCompressed) {
      return nullptr;
    }
    tfCompressed->decompress(tfDecoded);
    return tfDecoded;
  }
  return tfRaw;
}

void CMVFileHandle::close()
{
  if (tree) {
    tree->ResetBranchAddresses();
    tree = nullptr;
  }
  tfCompressed = nullptr;
  tfRaw = nullptr;
  delete tfDecoded;
  tfDecoded = nullptr;
  if (file) {
    file->Close();
    delete file;
    file = nullptr;
  }
}

} // namespace o2::tpc
