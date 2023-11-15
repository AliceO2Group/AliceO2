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

/// @file   VertexReaderSpec.h

#ifndef O2_ITS3_VERTEXREADER
#define O2_ITS3_VERTEXREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
namespace its3
{
// read ITS vertices from the output tree of ITS tracking

class VertexReader : public o2::framework::Task
{
  using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

 public:
  VertexReader() = default;
  ~VertexReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);
  void accumulate();

  std::vector<o2::itsmft::ROFRecord> mVerticesROFRec, *mVerticesROFRecPtr = &mVerticesROFRec;
  std::vector<Vertex> mVertices, *mVerticesPtr = &mVertices;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginITS;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mVertexTreeName = "o2sim";
  std::string mVertexBranchName = "Vertices";
  std::string mVertexROFBranchName = "VerticesROF";
};

/// create a processor spec
/// read ITS vertex data from a root file
o2::framework::DataProcessorSpec getITS3VertexReaderSpec();

} // namespace its3
} // namespace o2

#endif /* O2_ITS3_VERTEXREADER */
