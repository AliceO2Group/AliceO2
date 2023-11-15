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

/// @file   TrackReaderSpec.h

#ifndef O2_ITS3_TRACKREADER
#define O2_ITS3_TRACKREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace its3
{

class TrackReader : public o2::framework::Task
{
  using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

 public:
  TrackReader(bool useMC = true);
  ~TrackReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::itsmft::ROFRecord> mROFRec, *mROFRecInp = &mROFRec;
  std::vector<o2::itsmft::ROFRecord> mVerticesROFRec, *mVerticesROFRecInp = &mVerticesROFRec;
  std::vector<o2::its::TrackITS> mTracks, *mTracksInp = &mTracks;
  std::vector<Vertex> mVertices, *mVerticesInp = &mVertices;
  std::vector<int> mClusInd, *mClusIndInp = &mClusInd;
  std::vector<o2::MCCompLabel> mMCTruth, *mMCTruthInp = &mMCTruth;
  std::vector<o2::MCCompLabel> mMCVertTruth, *mMCVTruthInp = &mMCTruth;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginITS;

  bool mUseMC = true; // use MC truth

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInputFileName = "";
  std::string mTrackTreeName = "o2sim";
  std::string mROFBranchName = "IT3TracksROF";
  std::string mTrackBranchName = "IT3Track";
  std::string mClusIdxBranchName = "IT3TrackClusIdx";
  std::string mVertexBranchName = "Vertices";
  std::string mVertexROFBranchName = "VerticesROF";
  std::string mTrackMCTruthBranchName = "IT3TrackMCTruth";
  std::string mTrackMCVertTruthBranchName = "IT3VertexMCTruth";
};

/// create a processor spec
/// read ITS track data from a root file
framework::DataProcessorSpec getITS3TrackReaderSpec(bool useMC = true);

} // namespace its3
} // namespace o2

#endif /* O2_ITS3_TRACKREADER */
