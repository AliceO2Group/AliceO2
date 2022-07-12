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

#include "CommonUtils/DebugStreamer.h"
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include <thread>
#include <fmt/format.h>
#include "TROOT.h"
#include "TKey.h"
#endif

O2ParamImpl(o2::utils::ParameterDebugStreamer);

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && defined(DEBUG_STREAMER)

void o2::utils::DebugStreamer::setStreamer(const char* outFile, const char* option)
{
  if (!isStreamerSet()) {
    ROOT::EnableThreadSafety();
    mTreeStreamer = std::make_unique<o2::utils::TreeStreamRedirector>(fmt::format("{}_{}.root", outFile, getCPUID()).data(), option);
  }
}

std::string o2::utils::DebugStreamer::getUniqueTreeName(const char* tree) const { return fmt::format("{}_{}", tree, getNTrees()); }

size_t o2::utils::DebugStreamer::getCPUID() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }

int o2::utils::DebugStreamer::getNTrees() const { return mTreeStreamer->GetFile()->GetListOfKeys()->GetEntries(); }

void o2::utils::DebugStreamer::mergeTrees(const char* inpFile, const char* outFile, const char* option)
{
  TFile fInp(inpFile, "READ");
  TList list;
  for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
    const auto key = dynamic_cast<TKey*>(keyAsObj);
    list.Add((TTree*)fInp.Get(key->GetName()));
  }

  TFile fOut(outFile, "RECREATE");
  auto tree = TTree::MergeTrees(&list, option);
  fOut.WriteObject(tree, "tree");
}

void o2::utils::DebugStreamer::enableStream(const StreamFlags streamFlag)
{
  StreamFlags streamlevel = getStreamFlags();
  streamlevel = streamFlag | streamlevel;
  setStreamFlags(streamlevel);
}

void o2::utils::DebugStreamer::disableStream(const StreamFlags streamFlag)
{
  StreamFlags streamlevel = getStreamFlags();
  streamlevel = (~streamFlag) & streamlevel;
  setStreamFlags(streamlevel);
}

#endif
