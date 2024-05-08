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
#include <filesystem>
#include <sstream>

#include "MCHAlign/AlignRecordSpec.h"

#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "MathUtils/Utils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataTypes.h"
#include "Framework/TableBuilder.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/Task.h"
#include "FT0Base/Geometry.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackFitter.h"
#include "MCHTracking/TrackParam.h"
#include "MCHAlign/Aligner.h"
#include "MCHBase/TrackerParam.h"
#include "ForwardAlign/MillePedeRecord.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"

#include <TTree.h>
#include <TChain.h>

const int fgNCh = 10;
const int fgNDetElemCh[fgNCh] = {4, 4, 4, 4, 18, 18, 26, 26, 26, 26};
const int fgSNDetElemCh[fgNCh + 1] = {0, 4, 8, 12, 16, 34, 52, 78, 104, 130, 156};

namespace o2
{
namespace mch
{

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

using namespace std;
using std::cout;
using std::endl;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;

class AlignRecordTask
{
 public:
  //_________________________________________________________________________________________________
  AlignRecordTask(std::shared_ptr<DataRequest> dataRequest, std::shared_ptr<base::GRPGeomRequest> ccdbRequest, bool useMC = true)
    : mDataRequest(dataRequest), mCCDBRequest(ccdbRequest), mUseMC(useMC) {}

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {

    LOG(info) << "initializing align record maker";
    mTrackCount = 0;
    mTrackMatched = 0;
    if (mCCDBRequest) {
      base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    } else {
      auto grpFile = ic.options().get<std::string>("grp-file");
      if (std::filesystem::exists(grpFile)) {
        const auto grp = parameters::GRPObject::loadFrom(grpFile);
        base::Propagator::initFieldFromGRP(grp);
        TrackExtrap::setField();
        TrackExtrap::useExtrapV2();
        mAlign.SetBFieldOn(mch::TrackExtrap::isFieldON());
      } else {
        LOG(fatal) << "GRP file doesn't exist!";
      }
    }

    // Configuration for alignment object
    mAlign.SetDoEvaluation(false);
    mAlign.DisableRecordWriter();
    mAlign.SetAllowedVariation(0, 2.0);
    mAlign.SetAllowedVariation(1, 0.3);
    mAlign.SetAllowedVariation(2, 0.002);
    mAlign.SetAllowedVariation(3, 2.0);

    // Configuration for track fitter
    const auto& trackerParam = TrackerParam::Instance();
    trackFitter.setBendingVertexDispersion(trackerParam.bendingVertexDispersion);
    trackFitter.setChamberResolution(trackerParam.chamberResolutionX, trackerParam.chamberResolutionY);
    trackFitter.smoothTracks(true);
    trackFitter.useChamberResolution();
    mImproveCutChi2 = 2. * trackerParam.sigmaCutForImprovement * trackerParam.sigmaCutForImprovement;

    // Configuration for chamber fixing
    auto input_fixchambers = ic.options().get<string>("fix-chamber");
    std::stringstream string_chambers(input_fixchambers);
    string_chambers >> std::ws;
    while (string_chambers.good()) {
      string substr;
      std::getline(string_chambers, substr, ',');
      LOG(info) << Form("%s%d", "Fixing chamber: ", std::stoi(substr));
      mAlign.FixChamber(std::stoi(substr));
    }

    // Init for output saving
    auto OutputRecFileName = ic.options().get<string>("output-record-data");
    auto OutputConsFileName = ic.options().get<string>("output-record-constraint");
    mAlign.init(OutputRecFileName, OutputConsFileName);

    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>([this]() {
      LOG(info) << "Saving records into ROOT file";
      LOG(info) << "Nb of records to be saved: " << mTrackCount;
      LOG(info) << "Nb of matched MCH-MID tracks: " << mTrackMatched;
      mAlign.terminate();
    });
  }

  //_________________________________________________________________________________________________
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    /// finalize the track extrapolation setting
    if (mCCDBRequest && base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      if (matcher == framework::ConcreteDataMatcher("GLO", "GRPMAGFIELD", 0)) {
        TrackExtrap::setField();
        TrackExtrap::useExtrapV2();
        mAlign.SetBFieldOn(mch::TrackExtrap::isFieldON());
      }

      if (matcher == framework::ConcreteDataMatcher("GLO", "GEOMALIGN", 0)) {
        LOG(info) << "Loading reference geometry from CCDB";
        transformation = geo::transformationFromTGeoManager(*gGeoManager);
        for (int i = 0; i < 156; i++) {
          int iDEN = GetDetElemId(i);
          transform[iDEN] = transformation(iDEN);
        }
      }
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    if (mCCDBRequest) {
      base::GRPGeomHelper::instance().checkUpdates(pc);
    }

    o2::globaltracking::RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest.get());

    const auto& mchTracks = recoData.getMCHTracks();
    const auto& mchmidMatches = recoData.getMCHMIDMatches();
    const auto& mchClusters = recoData.getMCHTrackClusters();
    mTrackMatched += mchmidMatches.size();

    for (auto const& mchmidMatch : mchmidMatches) {

      int mchTrackID = mchmidMatch.getMCHRef().getIndex();
      const auto& mchTrack = mchTracks[mchTrackID];
      int first = mchTrack.getFirstClusterIdx();
      int last = mchTrack.getLastClusterIdx();
      int Ncluster = last - first + 1;

      if (Ncluster <= 9) {
        continue;
      }

      mch::Track convertedTrack;

      for (int i = first; i <= last; i++) {
        const auto& cluster = mchClusters[i];
        convertedTrack.createParamAtCluster(cluster);
      }

      // Erase removable track
      if (!RemoveTrack(convertedTrack)) {
        mAlign.ProcessTrack(convertedTrack, transformation, false, weightRecord);
        mTrackCount += 1;
        pc.outputs().snapshot(Output{"MUON", "RECORD_MCHMID", 0}, mAlign.GetRecord());
      }
    }
  }

 private:
  //_________________________________________________________________________________________________
  bool RemoveTrack(mch::Track& track)
  {
    bool removeTrack = false;

    try {
      trackFitter.fit(track, false);
    } catch (exception const& e) {
      removeTrack = true;
      return removeTrack;
    }

    auto itStartingParam = std::prev(track.rend());

    while (true) {
      try {
        trackFitter.fit(track, true, false, (itStartingParam == track.rbegin()) ? nullptr : &itStartingParam);
      } catch (exception const&) {
        removeTrack = true;
        break;
      }

      double worstLocalChi2 = -1.0;

      track.tagRemovableClusters(0x1F, false);
      auto itWorstParam = track.end();

      for (auto itParam = track.begin(); itParam != track.end(); ++itParam) {
        if (itParam->getLocalChi2() > worstLocalChi2) {
          worstLocalChi2 = itParam->getLocalChi2();
          itWorstParam = itParam;
        }
      }

      if (worstLocalChi2 < mImproveCutChi2) {
        break;
      }

      if (!itWorstParam->isRemovable()) {
        removeTrack = true;
        track.removable();
        break;
      }

      auto itNextParam = track.removeParamAtCluster(itWorstParam);
      auto itNextToNextParam = (itNextParam == track.end()) ? itNextParam : std::next(itNextParam);
      itStartingParam = track.rbegin();

      if (track.getNClusters() < 10) {
        removeTrack = true;
        break;
      } else {
        while (itNextToNextParam != track.end()) {
          if (itNextToNextParam->getClusterPtr()->getChamberId() != itNextParam->getClusterPtr()->getChamberId()) {
            itStartingParam = std::make_reverse_iterator(++itNextParam);
            break;
          }
          ++itNextToNextParam;
        }
      }
    }

    if (!removeTrack) {
      for (auto& param : track) {
        param.setParameters(param.getSmoothParameters());
        param.setCovariances(param.getSmoothCovariances());
      }
    }

    return removeTrack;
  }

  //_________________________________________________________________________________________________
  Int_t GetDetElemId(Int_t iDetElemNumber)
  {
    // make sure detector number is valid
    if (!(iDetElemNumber >= fgSNDetElemCh[0] &&
          iDetElemNumber < fgSNDetElemCh[fgNCh])) {
      LOG(fatal) << "Invalid detector element number: " << iDetElemNumber;
    }
    /// get det element number from ID
    // get chamber and element number in chamber
    int iCh = 0;
    int iDet = 0;
    for (int i = 1; i <= fgNCh; i++) {
      if (iDetElemNumber < fgSNDetElemCh[i]) {
        iCh = i;
        iDet = iDetElemNumber - fgSNDetElemCh[i - 1];
        break;
      }
    }

    // make sure detector index is valid
    if (!(iCh > 0 && iCh <= fgNCh && iDet < fgNDetElemCh[iCh - 1])) {
      LOG(fatal) << "Invalid detector element id: " << 100 * iCh + iDet;
    }

    // add number of detectors up to this chamber
    return 100 * iCh + iDet;
  }

  std::shared_ptr<base::GRPGeomRequest> mCCDBRequest; ///< pointer to the CCDB requests
  std::shared_ptr<DataRequest> mDataRequest;
  GID::mask_t mInputSources;
  bool mUseMC = true;
  parameters::GRPMagField* grpmag;
  TGeoManager* geo;

  mch::TrackFitter trackFitter;
  double mImproveCutChi2{};
  int mTrackCount{};
  int mTrackMatched{};
  mch::Aligner mAlign{};
  Double_t weightRecord{1.0};
  std::vector<o2::fwdalign::MillePedeRecord> mRecords;

  map<int, math_utils::Transform3D> transform;
  mch::geo::TransformationCreator transformation;
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getAlignRecordSpec(bool useMC, bool disableCCDB)
{
  auto dataRequest = std::make_shared<DataRequest>();
  o2::dataformats::GlobalTrackID::mask_t src = o2::dataformats::GlobalTrackID::getSourcesMask("MCH-MID");
  dataRequest->requestMCHClusters(false);
  dataRequest->requestTracks(src, useMC);

  vector<OutputSpec> outputSpecs{};
  auto ccdbRequest = disableCCDB ? nullptr : std::make_shared<base::GRPGeomRequest>(false,                         // orbitResetTime
                                                                                    false,                         // GRPECS=true
                                                                                    false,                         // GRPLHCIF
                                                                                    true,                          // GRPMagField
                                                                                    false,                         // askMatLUT
                                                                                    base::GRPGeomRequest::Aligned, // geometry
                                                                                    dataRequest->inputs,
                                                                                    true); // query only once all objects except mag.field

  outputSpecs.emplace_back("MUON", "RECORD_MCHMID", 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "mch-align-record",
    dataRequest->inputs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<AlignRecordTask>(dataRequest, ccdbRequest, useMC)},
    Options{{"geo-file", VariantType::String, o2::base::NameConf::getAlignedGeomFileName(), {"Name of the reference geometry file"}},
            {"grp-file", VariantType::String, o2::base::NameConf::getGRPFileName(), {"Name of the grp file"}},
            {"fix-chamber", VariantType::String, "", {"Chamber fixing, ex 1,2,3"}},
            {"output-record-data", VariantType::String, "recDataFile.root", {"Option for name of output record file for data"}},
            {"output-record-constraint", VariantType::String, "recConsFile.root", {"Option for name of output record file for constraint"}}}};
}

} // namespace mch
} // namespace o2