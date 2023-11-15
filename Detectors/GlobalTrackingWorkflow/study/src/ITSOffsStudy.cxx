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

#include "GlobalTrackingStudy/ITSOffsStudy.h"

#include <vector>
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonUtils/NameConf.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include <TH1F.h>

namespace o2::trackstudy
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class ITSOffsStudy : public Task
{
 public:
  ITSOffsStudy(std::shared_ptr<DataRequest> dr, GTrackID::mask_t src) : mDataRequest(dr), mTracksSrc(src) {}
  ~ITSOffsStudy() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  GTrackID::mask_t mTracksSrc{};
  std::unique_ptr<TH1F> mDTHisto{};
  const std::string mOutName{"its_offset_Study.root"};
};

void ITSOffsStudy::init(InitContext& ic)
{
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");
  mDTHisto = std::make_unique<TH1F>("dT", "T_{TOF} - T_{ITS-ROF}, #mus", 1000, 1, -1);
  mDTHisto->SetDirectory(nullptr);
}

void ITSOffsStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void ITSOffsStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
  }
}

void ITSOffsStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  constexpr float PS2MUS = 1e-6;
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto tofClusters = recoData.getTOFClusters();
  auto itsROFs = recoData.getITSTracksROFRecords();
  std::vector<int> itsTr2ROFID;
  std::unordered_map<GTrackID, char> ambigMap;

  int cntROF = 0;
  for (const auto& rof : itsROFs) {
    size_t maxE = rof.getFirstEntry() + rof.getNEntries();
    if (itsTr2ROFID.size() < maxE) {
      itsTr2ROFID.resize(maxE, cntROF);
    }
    cntROF++;
  }

  int nv = vtxRefs.size();
  for (int iv = 0; iv < nv; iv++) {
    const auto& vtref = vtxRefs[iv];
    for (int is = 0; is < GTrackID::NSources; is++) {
      if (!mTracksSrc[is] || !(GTrackID::getSourceDetectorsMask(is)[GTrackID::ITS] && GTrackID::getSourceDetectorsMask(is)[GTrackID::TOF])) {
        continue;
      }
      int idMin = vtxRefs[iv].getFirstEntryOfSource(is), idMax = idMin + vtxRefs[iv].getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        if (vid.isAmbiguous()) {
          auto& ambEntry = ambigMap[vid];
          if (ambEntry > 0) {
            continue;
          }
          ambEntry = 1;
        }
        auto tofMatch = recoData.getTOFMatch(vid);
        auto refs = recoData.getSingleDetectorRefs(vid);
        if (!refs[GTrackID::ITS].isIndexSet()) { // might be an afterburner track
          continue;
        }
        const auto& tofCl = tofClusters[refs[GTrackID::TOF]];
        float timeTOFMUS = (tofCl.getTime() - recoData.getTOFMatch(vid).getLTIntegralOut().getTOF(o2::track::PID::Pion)) * PS2MUS;

        int itsTrackID = refs[GTrackID::ITS].getIndex();
        // find the ROF corresponding to this track
        const auto& rof = itsROFs[itsTr2ROFID[itsTrackID]];
        auto tsROF = rof.getBCData().differenceInBC(recoData.startIR) * o2::constants::lhc::LHCBunchSpacingMUS;
        (*mDBGOut) << "itstof"
                   << "gid=" << vid << "ttof=" << timeTOFMUS << "tits=" << tsROF << "itsROFID=" << itsTr2ROFID[itsTrackID] << "\n";
        mDTHisto->Fill(timeTOFMUS - tsROF);
        const auto& trc = recoData.getTrackParam(tofMatch.getTrackRef());
        (*mDBGOut) << "dttof"
                   << "refgid=" << tofMatch.getTrackRef() << "dtime=" << tofMatch.getDeltaT() << "phi=" << trc.getPhi() << "tgl=" << trc.getTgl() << "q2t=" << trc.getQ2Pt() << "\n";
      }
    }
  }
}

void ITSOffsStudy::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
  TFile fout(mOutName.c_str(), "update");
  fout.WriteTObject(mDTHisto.get());
  LOGP(info, "Stored time differences histogram {} and tree {} into {}", mDTHisto->GetName(), "itstof", mOutName.c_str());
  fout.Close();
}

void ITSOffsStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
}

DataProcessorSpec getITSOffsStudy(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  bool useMC = false;
  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertertices(useMC);

  return DataProcessorSpec{
    "its-offset-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSOffsStudy>(dataRequest, srcTracks)},
    Options{}};
}

} // namespace o2::trackstudy
