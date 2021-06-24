// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/tracks-reader-workflow.cxx
/// \brief  DPL workflow to send MID tracks read from a root file
/// \author Philippe Pillot, Subatech

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"

#include "DPLUtils/RootTreeReader.h"

#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"

using namespace o2::framework;
using namespace o2::mid;

class TrackSamplerTask
{
 public:
  /// prepare the reader
  void init(InitContext& ic)
  {
    auto inputFileName = ic.options().get<std::string>("infile");
    mMinNumberOfROFsPerTF = ic.options().get<int>("repack-rofs");

    mReader = std::make_unique<RootTreeReader>("midreco", inputFileName.c_str(), -1,
                                               RootTreeReader::PublishingMode::Single,
                                               RootTreeReader::BranchDefinition<std::vector<char>>{
                                                 Output{"MID", "TRACKS", 0, Lifetime::Timeframe}, "MIDTrack"},
                                               RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{
                                                 Output{"MID", "TRACKSROF", 0, Lifetime::Timeframe}, "MIDTrackROF"},
                                               &mAccumulator);
  }

  /// process the next entry
  void run(ProcessingContext& pc)
  {
    if (mReader->next()) {
      (*mReader)(pc);
      if (mROFs.size() >= mMinNumberOfROFsPerTF) {
        publish(pc.outputs());
      }
    } else {
      if (mROFs.size() > 0) {
        publish(pc.outputs());
      }
      pc.services().get<ControlService>().endOfStream();
    }
  }

 private:
  /// accumulate the data
  bool accumulate(std::string_view name, char* data)
  {
    if (name == "MIDTrackROF") {

      auto rofs = reinterpret_cast<std::vector<ROFRecord>*>(data);

      // accumulate the ROFs, shifting the track indexing accordingly
      size_t offset = (mROFs.size() > 0) ? mROFs.back().firstEntry + mROFs.back().nEntries : 0;
      std::transform(rofs->begin(), rofs->end(), std::back_inserter(mROFs), [offset](const ROFRecord& rof) {
        return ROFRecord{rof, rof.firstEntry + offset, rof.nEntries};
      });

    } else if (name == "MIDTrack") {

      auto tracks = reinterpret_cast<std::vector<char>*>(data);

      // accumulate the tracks, in Track format
      if (tracks->size() % sizeof(Track) != 0) {
        throw std::length_error("invalid track format");
      }
      size_t offset = mTracks.size();
      mTracks.resize(offset + tracks->size() / sizeof(Track));
      std::memcpy(&(mTracks[offset]), tracks->data(), tracks->size());

    } else {
      throw std::invalid_argument("invalid branch");
    }

    return true;
  }

  /// publish the data and clear the internal vector
  void publish(DataAllocator& out)
  {
    out.snapshot(OutputRef{"rofs"}, mROFs);
    out.snapshot(OutputRef{"tracks"}, mTracks);
    mROFs.clear();
    mTracks.clear();
  }

  size_t mMinNumberOfROFsPerTF = 1;          ///< minimum number of ROF to send per TF
  std::vector<ROFRecord> mROFs{};            ///< internal vector of ROFs
  std::vector<Track> mTracks{};              ///< internal vector of tracks
  std::unique_ptr<RootTreeReader> mReader{}; ///< root file reader
  /// structure holding the function to accumulate the data
  RootTreeReader::SpecialPublishHook mAccumulator{
    [this](std::string_view name, ProcessingContext&, Output const&, char* data) -> bool {
      return this->accumulate(name, data);
    }};
};

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  return WorkflowSpec{DataProcessorSpec{
    "TrackSampler",
    Inputs{},
    Outputs{OutputSpec{{"rofs"}, "MID", "TRACKSROF", 0, Lifetime::Timeframe},
            OutputSpec{{"tracks"}, "MID", "TRACKS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackSamplerTask>()},
    Options{{"infile", VariantType::String, "mid-reco.root", {"input filename"}},
            {"repack-rofs", VariantType::Int, 1, {"min number of rofs per timeframe"}}}}};
}
