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

/// @file   CMVHelper.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Helper utilities for reading CMV ROOT files

#ifndef ALICEO2_TPC_CMVHELPER_H_
#define ALICEO2_TPC_CMVHELPER_H_

#include <string>

#include "TFile.h"
#include "TTree.h"

namespace o2::tpc
{

struct CMVPerTF;
struct CMVPerTFCompressed;

struct CMVFileHandle {
  TFile* file{nullptr};
  TTree* tree{nullptr};
  bool isCompressed{false};
  CMVPerTFCompressed* tfCompressed{nullptr};
  CMVPerTF* tfRaw{nullptr};
  CMVPerTF* tfDecoded{nullptr}; ///< scratch buffer used when decompressing
  long firstTFInTree{-1};       ///< first global TF index from tree UserInfo ("firstTF"); -1 if absent
  long lastTFInTree{-1};        ///< last  global TF index from tree UserInfo ("lastTF");  -1 if absent

  /// Open path and set up branch addresses. Returns false on any error
  bool open(const std::string& path);

  /// Load entry iEntry and return a pointer to the decoded CMVPerTF, or nullptr on error
  const CMVPerTF* getEntry(long long iEntry);

  /// Release all resources
  void close();
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVHELPER_H_
