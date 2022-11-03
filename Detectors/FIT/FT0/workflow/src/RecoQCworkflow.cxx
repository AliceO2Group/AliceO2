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

/// \file   RecoQCworkflow.cxx
///\ brief QC for  reconstructed data
/// \author Alla.Maevskaya@cern.ch

#include <fairlogger/Logger.h>
#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "FT0Workflow/RecoQCworkflow.h"
#include "TStopwatch.h"
#include <TString.h>
#include <vector>
#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>
#include <TMath.h>
#include <vector>
#include <map>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;

namespace o2::ft0
{

void RecoQCworkflow::init(InitContext& ic)
{
  mFileOut = TFile::Open("RecoQChistos.root", "RECREATE");
  TString histnames[9] = {"hT0AcorrPV", "hT0CcorrPV", "resolution", "hT0A", "hT0C", "hT0AC"};
  for (int ihist = 0; ihist < 6; ihist++) {
    mHisto[ihist] = new TH1F(histnames[ihist].Data(), histnames[ihist].Data(), 300, -1000, 1000);
  }
  mVertexT0 = new TH1F("VertexT0", "T0 vertex", 100, -30, 30);
  mPV = new TH1F("PV", "primary vertex", 100, -30, 30);
  mVertexComp = new TH2F("hVertexComp", "FT0 and PV comparion", 100, -30, 30, 100, -30, 30);
  mTimer.Stop();
  mTimer.Reset();
}
void RecoQCworkflow::run(o2::framework::ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  auto primVertices = recoData.getPrimaryVertices();
  auto ft0RecPoints = recoData.getFT0RecPoints();
  std::map<uint64_t, o2::dataformats::PrimaryVertex const*> bcsMap;
  for (auto& vertex : primVertices) {
    auto& timeStamp = vertex.getTimeStamp();
    double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    auto [iter, inserted] = bcsMap.try_emplace(globalBC, &vertex);
    if (!inserted) {
      iter->second = nullptr;
    }
  }
  float vertexT0;
  for (auto& ft0RecPoint : ft0RecPoints) {
    uint64_t bc = ft0RecPoint.getInteractionRecord().toLong();
    auto item = bcsMap.find(bc);

    if (std::abs(ft0RecPoint.getCollisionTimeA()) < 2000) {
      mHisto[3]->Fill(ft0RecPoint.getCollisionTimeA());
    }
    if (std::abs(ft0RecPoint.getCollisionTimeC()) < 2000) {
      mHisto[4]->Fill(ft0RecPoint.getCollisionTimeC());
    }

    if (std::abs(ft0RecPoint.getCollisionTimeC()) < 2000 &&
        std::abs(ft0RecPoint.getCollisionTimeA()) < 2000) {
      mHisto[5]->Fill(ft0RecPoint.getCollisionTimeMean());
      vertexT0 = 0.5 * (ft0RecPoint.getCollisionTimeC() - ft0RecPoint.getCollisionTimeA()) * cSpeed;
      mVertexT0->Fill(vertexT0);
    }
    if (item == bcsMap.end() || item->second == nullptr) {
      LOG(debug) << "Error: could not find a corresponding BC ID for a FT0 rec. point; BC = " << bc;
      continue;
    }
    auto& vertex = *item->second;
    auto currentVertex = vertex.getZ();
    mPV->Fill(currentVertex);
    ushort ncont = vertex.getNContributors();
    LOG(debug) << "CurrentVertex " << currentVertex << " ncont " << int(ncont);
    if (ncont < 3) {
      continue;
    }
    auto shift = currentVertex / cSpeed;
    short t0A = ft0RecPoint.getCollisionTimeA() + shift;
    short t0C = ft0RecPoint.getCollisionTimeC() - shift;

    LOG(info) << " BC  t0  " << bc << " shift " << shift << " A " << t0A << " C " << t0C << " vertex " << vertexT0 << " PV " << currentVertex;
    mHisto[0]->Fill(t0A);
    mHisto[1]->Fill(t0C);
    mHisto[2]->Fill((t0C - t0A) / 2);
    mVertexComp->Fill(vertexT0, currentVertex);
  }
  mTimer.Stop();
}
void RecoQCworkflow::endOfStream(EndOfStreamContext& ec)
{
  mFileOut->cd();
  for (int ihist = 0; ihist < 6; ihist++) {
    mHisto[ihist]->Write();
  }
  mPV->Write();
  mVertexT0->Write();
  mVertexComp->Write();
  mFileOut->Close();
}

DataProcessorSpec getRecoQCworkflow(GID::mask_t src)
{
  auto dataRequest = std::make_shared<DataRequest>();
  LOG(info) << "@@ request primary vertex";
  dataRequest->requestPrimaryVertertices(false);
  dataRequest->requestFT0RecPoints(false);
  LOG(info) << "@@@ requested T0";
  std::vector<OutputSpec> outputs; // empty

  return DataProcessorSpec{
    "reco-qc",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::ft0::RecoQCworkflow>(src, dataRequest)},
    Options{}};
}

}; // namespace o2::ft0
