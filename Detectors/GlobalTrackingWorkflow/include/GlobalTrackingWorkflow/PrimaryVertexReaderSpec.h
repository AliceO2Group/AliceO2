// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   PrimaryVertexReaderSpec.h

#ifndef O2_PRIMARY_VERTEXREADER
#define O2_PRIMARY_VERTEXREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "SimulationDataFormat/MCEventLabel.h"

namespace o2
{
namespace vertexing
{
// read primary vertices produces by the o2-primary-vertexing-workflow

class PrimaryVertexReader : public o2::framework::Task
{
  using Label = o2::MCEventLabel;
  using V2TRef = o2::dataformats::VtxTrackRef;
  using PVertex = o2::dataformats::PrimaryVertex;
  using GIndex = o2::dataformats::VtxTrackIndex;

 public:
  PrimaryVertexReader(bool useMC) : mUseMC(useMC) {}
  ~PrimaryVertexReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree();

  bool mVerbose = false;
  bool mUseMC = false;

  std::vector<PVertex> mVertices, *mVerticesPtr = &mVertices;
  std::vector<Label> mLabels, *mLabelsPtr = &mLabels;
  std::vector<V2TRef> mPV2MatchIdxRef, *mPV2MatchIdxRefPtr = &mPV2MatchIdxRef;
  std::vector<GIndex> mPV2MatchIdx, *mPV2MatchIdxPtr = &mPV2MatchIdx;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mFileNameMatches = "";
  std::string mVertexTreeName = "o2sim";
  std::string mVertexBranchName = "PrimaryVertex";
  std::string mVertexTrackIDsBranchName = "PVTrackIndices";
  std::string mVertex2TrackIDRefsBranchName = "PV2TrackRefs";
  std::string mVertexLabelsBranchName = "PVMCTruth";
};

/// create a processor spec
/// read primary vertex data from a root file
o2::framework::DataProcessorSpec getPrimaryVertexReaderSpec(bool useMC);

} // namespace vertexing
} // namespace o2

#endif
