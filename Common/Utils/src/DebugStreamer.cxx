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
#include <random>
#include "Framework/Logger.h"
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

o2::utils::TreeStreamRedirector& o2::utils::DebugStreamer::getStreamer(const char* outFile, const char* option, const size_t id)
{
  setStreamer(outFile, option, id);
  return getStreamer(id);
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

bool o2::utils::DebugStreamer::checkStream(const StreamFlags streamFlag, const size_t samplingID)
{
  const bool isStreamerSet = ((getStreamFlags() & streamFlag) == streamFlag);
  if (!isStreamerSet) {
    return false;
  }

  // check sampling frequency
  const auto sampling = getSamplingTypeFrequency(streamFlag);
  if (sampling.first != SamplingTypes::sampleAll) {
    // init random number generator for each thread
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<> distr(0, 1);

    auto sampleTrack = [&]() {
      if (samplingID == -1) {
        LOGP(fatal, "Sampling type sampleID not supported for stream flag {}", streamFlag);
      }
      std::uniform_real_distribution<> distr(0, 1);
      // sample on samplingID (e.g. track level)
      static thread_local std::unordered_map<StreamFlags, std::pair<size_t, bool>> idMap;
      // in case of first call samplingID in idMap is 0 and always false and first ID rejected
      if (idMap[streamFlag].first != samplingID) {
        idMap[streamFlag] = std::pair<size_t, bool>{samplingID, (distr(generator) < sampling.second)};
      }
      return idMap[streamFlag].second;
    };

    if (sampling.first == SamplingTypes::sampleRandom) {
      // just sample randomly
      return (distr(generator) < sampling.second);
    } else if (sampling.first == SamplingTypes::sampleID) {
      return sampleTrack();
    } else if (sampling.first == SamplingTypes::sampleIDGlobal) {
      // this contains for each flag the processed track IDs and stores if it was processed or not
      static tbb::concurrent_unordered_map<int, tbb::concurrent_unordered_map<size_t, bool>> refIDs;
      const int index = ParameterDebugStreamer::Instance().sampleIDGlobal[getIndex(streamFlag)];

      // check if refIDs contains track ID
      auto it = refIDs[index].find(samplingID);
      if (it != refIDs[index].end()) {
        // in case it is present get stored decission
        return it->second;
      } else {
        // in case it is not present sample random decission
        const bool storeTrk = sampleTrack();
        refIDs[index][samplingID] = storeTrk;
        return storeTrk;
      }
    }
  }
  return true;
}

int o2::utils::DebugStreamer::getIndex(const StreamFlags streamFlag)
{
  // see: https://stackoverflow.com/a/71539401
  uint32_t v = streamFlag;
  v -= 1;
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  const uint32_t ind = (((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24);
  return ind;
}

std::pair<o2::utils::SamplingTypes, float> o2::utils::DebugStreamer::getSamplingTypeFrequency(const StreamFlags streamFlag)
{
  const int ind = getIndex(streamFlag);
  return std::pair<o2::utils::SamplingTypes, float>{ParameterDebugStreamer::Instance().samplingType[ind], ParameterDebugStreamer::Instance().samplingFrequency[ind]};
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
