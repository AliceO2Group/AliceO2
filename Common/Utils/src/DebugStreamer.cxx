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

o2::utils::DebugStreamer::DebugStreamer()
{
  ROOT::EnableThreadSafety();
}

void o2::utils::DebugStreamer::setStreamer(const char* outFile, const char* option, const size_t id)
{
  if (!isStreamerSet(id)) {
    mTreeStreamer[id] = std::make_unique<o2::utils::TreeStreamRedirector>(fmt::format("{}_{}.root", outFile, id).data(), option);
  }
}

void o2::utils::DebugStreamer::flush(const size_t id)
{
  if (isStreamerSet(id)) {
    mTreeStreamer[id].reset();
  }
}

void o2::utils::DebugStreamer::flush()
{
  for (const auto& pair : mTreeStreamer) {
    flush(pair.first);
  }
}

std::string o2::utils::DebugStreamer::getUniqueTreeName(const char* tree, const size_t id) const { return fmt::format("{}_{}", tree, getNTrees(id)); }

size_t o2::utils::DebugStreamer::getCPUID() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }

o2::utils::TreeStreamRedirector* o2::utils::DebugStreamer::getStreamerPtr(const size_t id) const
{
  auto it = mTreeStreamer.find(id);
  if (it != mTreeStreamer.end()) {
    return (it->second).get();
  } else {
    return nullptr;
  }
}

int o2::utils::DebugStreamer::getNTrees(const size_t id) const { return isStreamerSet(id) ? getStreamerPtr(id)->GetFile()->GetListOfKeys()->GetEntries() : -1; }

void o2::utils::DebugStreamer::mergeTrees(const char* inpFile, const char* outFile, const char* option)
{
  TFile fInp(inpFile, "READ");
  std::unordered_map<int, TList> lists;
  for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
    const auto key = dynamic_cast<TKey*>(keyAsObj);
    TTree* tree = (TTree*)fInp.Get(key->GetName());
    // perform simple check on the number of entries to merge only TTree with same content (ToDo: Do check on name of branches)
    const int entries = tree->GetListOfBranches()->GetEntries();
    lists[entries].Add(tree);
  }

  TFile fOut(outFile, "RECREATE");
  for (auto& list : lists) {
    auto tree = TTree::MergeTrees(&list.second, option);
    fOut.WriteObject(tree, tree->GetName());
  }
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
