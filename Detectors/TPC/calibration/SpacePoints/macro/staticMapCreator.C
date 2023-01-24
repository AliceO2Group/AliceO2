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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"

#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/TrackInterpolation.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsTPC/Defs.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"

#include <TFile.h>
#include <TTree.h>
#include <TGeoManager.h>
#include <TGrid.h>

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <array>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#else

#error This macro must run in compiled mode

#endif

using namespace o2::tpc;
using GID = o2::dataformats::GlobalTrackID;

std::vector<string> getInputFileList(const std::string& fileInput)
{
  std::vector<std::string> fileList;
  std::vector<std::string> fileListVerified;
  // check if only one input file (a txt file contaning a list of files is provided)
  if (boost::algorithm::ends_with(fileInput, "txt")) {
    LOGP(info, "Reading files from input file list {}", fileInput);
    std::ifstream is(fileInput);
    std::istream_iterator<std::string> start(is);
    std::istream_iterator<std::string> end;
    fileList.insert(fileList.begin(), start, end);
  } else {
    fileList.push_back(fileInput);
  }

  for (const auto& file : fileList) {
    if ((file.find("alien://") == 0) && !gGrid && !TGrid::Connect("alien://")) {
      LOG(fatal) << "Failed to open alien connection";
    }
    std::unique_ptr<TFile> filePtr(TFile::Open(file.data()));
    if (!filePtr || !filePtr->IsOpen() || filePtr->IsZombie()) {
      LOGP(warning, "Could not open file {}", file);
      continue;
    }
    fileListVerified.push_back(file);
  }

  if (fileListVerified.size() == 0) {
    LOGP(error, "No input files to process");
  }
  return fileListVerified;
}

bool revalidateTrack(const TrackData& trk, const SpacePointsCalibConfParam& params)
{
  if (trk.nClsITS < params.minITSNCls) {
    return false;
  }
  if (trk.nClsTPC < params.minTPCNCls) {
    return false;
  }
  if (trk.nTrkltsTRD > 0 && trk.nTrkltsTRD < params.minTRDNTrklts) {
    // in case nTrkltsTRD == 0 this is an ITS-TPC-TOF track which we might not want to cut
    return false;
  }
  // track quality cuts
  if (trk.chi2ITS / trk.nClsITS > params.maxITSChi2) {
    return false;
  }
  if (trk.chi2TPC / trk.nClsTPC > params.maxTPCChi2) {
    return false;
  }
  if (trk.nTrkltsTRD > 0 && trk.chi2TRD / trk.nTrkltsTRD > params.maxTRDChi2) {
    return false;
  }

  if (params.cutOnDCA) {
    auto propagator = o2::base::Propagator::Instance();
    // o2::track::TrackPar trkPar(trk.x, trk.alpha, trk.p); // use this line, in case ClassDef version of TrackData < 4
    o2::track::TrackPar trkPar = trk.par;
    if (!propagator->propagateToX(trkPar, 0, propagator->getNominalBz())) {
      return false;
    }
    if (trkPar.getX() * trkPar.getX() + trkPar.getY() * trkPar.getY() > params.maxDCA * params.maxDCA) {
      LOGP(debug, "DCA cut not passed {}", std::sqrt(trkPar.getX() * trkPar.getX() + trkPar.getY() * trkPar.getY()));
      return false;
    }
    LOGP(debug, "DCA cut OK {}", std::sqrt(trkPar.getX() * trkPar.getX() + trkPar.getY() * trkPar.getY()));
  }
  return true;
}

void staticMapCreator(std::string fileInput = "files.txt",
                      int runNumber = 527976,
                      std::string fileOutput = "voxRes.root",
                      std::string voxMapInput = "",
                      std::string trackSources = static_cast<std::string>(GID::ALL))
{

  // Obtain configuration
  const SpacePointsCalibConfParam& params = SpacePointsCalibConfParam::Instance();
  if (!boost::filesystem::exists("scdconfig.ini")) {
    LOG(warn) << "Did not find configuration file. Using default parameters and storing them in scdconfig.ini";
    params.writeINI("scdconfig.ini", "scdcalib"); // to write default parameters to a file
  } else {
    params.updateFromFile("scdconfig.ini");
  }
  LOG(info) << "----- Dumping configuration values START -----";
  params.printKeyValues();
  LOG(info) << "----- Dumping configuration values END -----";

  GID::mask_t allowedSources = GID::getSourcesMask("ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  GID::mask_t sources = allowedSources & GID::getSourcesMask(trackSources);

  // Get CCDB objects
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setURL("https://alice-ccdb.cern.ch");
  auto runDuration = ccdbmgr.getRunDuration(runNumber);
  auto tRun = runDuration.first + (runDuration.second - runDuration.first) / 2; // time stamp for the middle of the run duration
  ccdbmgr.setTimestamp(tRun);

  // CTP orbit reset time
  auto orbitResetTimeNS = ccdbmgr.get<std::vector<int64_t>>("CTP/Calib/OrbitReset");
  int64_t orbitResetTimeMS = (*orbitResetTimeNS)[0] * 1e-3;
  LOGP(info, "Orbit reset time in MS is {}", orbitResetTimeMS);

  auto geoAligned = ccdbmgr.get<TGeoManager>("GLO/Config/GeometryAligned");
  auto magField = ccdbmgr.get<o2::parameters::GRPMagField>("GLO/Config/GRPMagField");
  const o2::base::MatLayerCylSet* matLut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdbmgr.get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));
  o2::base::Propagator::initFieldFromGRP(magField);
  auto prop = o2::base::Propagator::Instance();
  prop->setMatLUT(matLut);

  // Input
  auto fileList = getInputFileList(fileInput);

  std::array<std::vector<TrackResiduals::LocalResid>, SECTORSPERSIDE * SIDES> binnedResidualsSec;     // binned residuals generated on-the-fly
  std::array<std::vector<TrackResiduals::LocalResid>*, SECTORSPERSIDE * SIDES> binnedResidualsSecPtr; // for setting branch addresses
  std::array<std::vector<TrackResiduals::VoxStats>, SECTORSPERSIDE * SIDES> voxStatsSec;              // voxel statistics generated on-the-fly
  std::vector<TrackResiduals::LocalResid> binnedResiduals, *binnedResidualsPtr = &binnedResiduals;    // binned residuals

  TrackResiduals trackResiduals;
  trackResiduals.init();

  // Do we have a correction map available that we should apply to the clusters before the map extraction?
  std::array<std::vector<TrackResiduals::VoxRes>, SECTORSPERSIDE * SIDES> voxelResults{};
  if (voxMapInput.size()) {
    LOG(info) << "A correction map has been provided. Will apply the corrections to the cluster residuals";
    for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
      voxelResults[iSec].resize(trackResiduals.getNVoxelsPerSector());
    }
    TrackResiduals::VoxRes voxRes, *voxResPtr = &voxRes;
    std::unique_ptr<TFile> fIn = std::make_unique<TFile>(voxMapInput.c_str());
    std::unique_ptr<TTree> treeIn;
    treeIn.reset((TTree*)fIn->Get("voxResTree"));
    treeIn->SetBranchAddress("voxRes", &voxResPtr);
    for (int iEntry = 0; iEntry < treeIn->GetEntries(); ++iEntry) {
      treeIn->GetEntry(iEntry);
      voxelResults[voxRes.bsec][trackResiduals.getGlbVoxBin(voxRes.bvox)] = voxRes;
    }
  }

  trackResiduals.createOutputFile(fileOutput.c_str());
  std::unique_ptr<TTree> treeBinnedResiduals = std::make_unique<TTree>("resid", "TPC binned residuals");
  if (!params.writeBinnedResiduals) {
    treeBinnedResiduals->SetDirectory(nullptr);
  }
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    binnedResidualsSecPtr[iSec] = &binnedResidualsSec[iSec];
    voxStatsSec[iSec].resize(trackResiduals.getNVoxelsPerSector());
    for (int ix = 0; ix < trackResiduals.getNXBins(); ++ix) {
      for (int ip = 0; ip < trackResiduals.getNY2XBins(); ++ip) {
        for (int iz = 0; iz < trackResiduals.getNZ2XBins(); ++iz) {
          auto& statsVoxel = voxStatsSec[iSec][trackResiduals.getGlbVoxBin(ix, ip, iz)];
          // COG estimates are set to the bin center by default
          trackResiduals.getVoxelCoordinates(iSec, ix, ip, iz, statsVoxel.meanPos[TrackResiduals::VoxX], statsVoxel.meanPos[TrackResiduals::VoxF], statsVoxel.meanPos[TrackResiduals::VoxZ]);
        }
      }
    }
    treeBinnedResiduals->Branch(Form("sec%d", iSec), &binnedResidualsSecPtr[iSec]);
  }

  std::unique_ptr<TFile> inputFile;
  std::unique_ptr<TTree> treeUnbinnedResiduals;
  std::unique_ptr<TTree> treeTrackData;
  std::unique_ptr<TTree> treeRecords;
  std::vector<UnbinnedResid> unbinnedResiduals, *unbinnedResidualsPtr = &unbinnedResiduals; // unbinned residuals input
  std::vector<TrackDataCompact> trackRefs, *trackRefsPtr = &trackRefs;                      // the track references for unbinned residuals
  std::vector<TrackData> trackData, *trackDataPtr = &trackData;                             // additional track information (chi2, nClusters, track parameters)
  std::vector<uint32_t> orbits, *orbitsPtr = &orbits;                                       // first orbit for each TF in the input data

  for (const auto& fileName : fileList) {
    LOGP(info, "Processing input file {}", fileName);
    treeUnbinnedResiduals.reset(nullptr);
    treeTrackData.reset(nullptr);
    treeRecords.reset(nullptr);
    inputFile.reset(TFile::Open(fileName.c_str()));
    if (!inputFile || inputFile->IsZombie()) {
      LOGP(info, "Skipping file {}", fileName);
      continue;
    }
    treeUnbinnedResiduals.reset((TTree*)inputFile->Get("unbinnedResid"));
    treeUnbinnedResiduals->SetBranchAddress("res", &unbinnedResidualsPtr);
    treeUnbinnedResiduals->SetBranchAddress("trackInfo", &trackRefsPtr);
    if (params.useTrackData) {
      treeTrackData.reset((TTree*)inputFile->Get("trackData"));
      treeTrackData->SetBranchAddress("trk", &trackDataPtr);
      if (treeTrackData->GetEntries() != treeUnbinnedResiduals->GetEntries()) {
        LOGP(error, "The input trees with unbinned residuals and track information have a different number of entries ({} vs {})",
             treeUnbinnedResiduals->GetEntries(), treeTrackData->GetEntries());
      }
    }
    treeRecords.reset((TTree*)inputFile->Get("records"));
    treeRecords->SetBranchAddress("firstTForbit", &orbitsPtr);
    treeRecords->GetEntry(0); // per input file there is only a single entry in the tree
    for (int iEntry = 0; iEntry < treeUnbinnedResiduals->GetEntries(); ++iEntry) {
      if (params.timeFilter) {
        int64_t tfTimeInMS = orbitResetTimeMS + orbits[iEntry] * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
        if (tfTimeInMS < params.startTimeMS || tfTimeInMS > params.endTimeMS) {
          LOGP(debug, "Dropping TF at index {} with time {} and orbit {}", iEntry, tfTimeInMS, orbits[iEntry]);
          continue;
        }
      }
      treeUnbinnedResiduals->GetEntry(iEntry);
      if (params.useTrackData) {
        treeTrackData->GetEntry(iEntry);
      }
      auto nTracks = trackRefs.size();
      for (size_t iTrack = 0; iTrack < nTracks; ++iTrack) {
        const auto& trkInfo = trackRefs[iTrack];
        if (!GID::includesSource(trkInfo.sourceId, sources)) {
          continue;
        }
        if (params.useTrackData) {
          const auto& trk = trackData[iTrack];
          if (!revalidateTrack(trk, params)) {
            continue;
          }
        }
        for (unsigned int i = trkInfo.idxFirstResidual; i < trkInfo.idxFirstResidual + trkInfo.nResiduals; ++i) {
          const auto& residIn = unbinnedResiduals[i];
          int sec = residIn.sec;
          auto& residVecOut = binnedResidualsSec[sec];
          auto& statVecOut = voxStatsSec[sec];
          std::array<unsigned char, TrackResiduals::VoxDim> bvox;
          float xPos = param::RowX[residIn.row];
          float yPos = residIn.y * param::MaxY / 0x7fff + residIn.dy * param::MaxResid / 0x7fff;
          float zPos = residIn.z * param::MaxZ / 0x7fff + residIn.dz * param::MaxResid / 0x7fff;
          if (!trackResiduals.findVoxelBin(sec, xPos, yPos, zPos, bvox)) {
            // we are not inside any voxel
            LOGF(debug, "Dropping residual in sec(%i), x(%f), y(%f), z(%f)", sec, xPos, yPos, zPos);
            continue;
          }
          if (voxMapInput.size()) {
            // we already have a correction map available which we want to apply as consistency check
            const auto& voxRes = voxelResults[sec][trackResiduals.getGlbVoxBin(bvox)];
            float dy = residIn.dy * param::MaxResid / 0x7fff;
            float tgSlp = residIn.tgSlp * param::MaxTgSlp / 0x7fff;
            dy -= voxRes.D[TrackResiduals::ResY] - voxRes.D[TrackResiduals::ResX] * tgSlp;
            float dz = residIn.dz * param::MaxResid / 0x7fff;
            dz -= voxRes.D[TrackResiduals::ResZ] - voxRes.D[TrackResiduals::ResX] * voxRes.stat[TrackResiduals::VoxZ];
            dy = fabs(dy) < param::MaxResid ? dy : std::copysign(param::MaxResid, dy);
            dz = fabs(dz) < param::MaxResid ? dz : std::copysign(param::MaxResid, dz);
            short dYcorr = static_cast<short>(dy * 0x7fff / param::MaxResid);
            short dZcorr = static_cast<short>(dz * 0x7fff / param::MaxResid);
            residVecOut.emplace_back(dYcorr, dZcorr, residIn.tgSlp, bvox);
          } else {
            residVecOut.emplace_back(residIn.dy, residIn.dz, residIn.tgSlp, bvox);
          }
          auto& stat = statVecOut[trackResiduals.getGlbVoxBin(bvox)];
          float& binEntries = stat.nEntries;
          float oldEntries = binEntries++;
          float norm = 1.f / binEntries;
          // update COG for voxel bvox (update for X only needed in case binning is not per pad row)
          float xPosInv = 1.f / xPos;
          stat.meanPos[TrackResiduals::VoxX] = (stat.meanPos[TrackResiduals::VoxX] * oldEntries + xPos) * norm;
          stat.meanPos[TrackResiduals::VoxF] = (stat.meanPos[TrackResiduals::VoxF] * oldEntries + yPos * xPosInv) * norm;
          stat.meanPos[TrackResiduals::VoxZ] = (stat.meanPos[TrackResiduals::VoxZ] * oldEntries + zPos * xPosInv) * norm;
        }
      }
    }
    treeBinnedResiduals->Fill();
    for (auto& resid : binnedResidualsSec) {
      resid.clear();
    }
  }

  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    // for each sector fill the vector of local residuals from the respective branch
    auto brResid = treeBinnedResiduals->GetBranch(Form("sec%d", iSec));
    brResid->SetAddress(&binnedResidualsPtr);
    for (int iEntry = 0; iEntry < brResid->GetEntries(); ++iEntry) {
      brResid->GetEntry(iEntry);
      trackResiduals.getLocalResVec().insert(trackResiduals.getLocalResVec().end(), binnedResiduals.begin(), binnedResiduals.end());
    }
    trackResiduals.setStats(voxStatsSec[iSec], iSec);
    // do processing
    trackResiduals.processSectorResiduals(iSec);
    // do cleanup
    trackResiduals.clear();
  }

  if (params.writeBinnedResiduals) {
    trackResiduals.getOutputFilePtr()->cd();
    treeBinnedResiduals->Write();
  }
  treeBinnedResiduals.reset();
  trackResiduals.closeOutputFile();

  treeUnbinnedResiduals.reset();
  treeTrackData.reset();
  treeRecords.reset();
  inputFile.reset();

  LOG(info) << "Done processing";
}
