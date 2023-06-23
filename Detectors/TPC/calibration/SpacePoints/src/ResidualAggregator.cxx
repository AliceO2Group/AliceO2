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

/// \file ResidualAggregator.cxx
/// \brief Collects local TPC cluster residuals from EPNs
/// \author Ole Schmidt

#include "SpacePoints/ResidualAggregator.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"

#include <filesystem>
#include <numeric>
#include <algorithm>
#include <chrono>

using namespace o2::tpc;

ResidualsContainer::~ResidualsContainer()
{
  // trees must be deleted before the file is closed, otherwise segfaults
  LOGP(debug, "Deleting ResidualsContainer with {} entries. File name is {}", getNEntries(), fileName);
  treeOutResidualsUnbinned.reset();
  treeOutTrackData.reset();
  treeOutResiduals.reset();
  treeOutStats.reset();
  treeOutRecords.reset();
  if (fileOut) {
    LOG(debug) << "Removing output file of discarded slot";
    // this slot was not finalized, need to close and remove the file
    fileOut->Close();
    fileOut.reset();
    std::string fileToRemove = fileName + ".part";
    if (!std::filesystem::remove(fileToRemove)) {
      LOG(warning) << "Tried to delete, but could not find file named " << fileToRemove;
    }
  }
}

ResidualsContainer::ResidualsContainer(const ResidualsContainer& rhs)
{
  LOG(error) << "Must not call copy constructor of ResidualsContainer";
  // the copy constructor is needed because of ROOT's ClassDef
  // but should never actually be called
}

ResidualsContainer::ResidualsContainer(ResidualsContainer&& rhs)
{
  LOGP(debug, "Move operator called for rhs with {} entries", rhs.nResidualsTotal);
  trackResiduals = rhs.trackResiduals;
  fileOut = std::move(rhs.fileOut);
  fileName = std::move(rhs.fileName);
  treeOutResidualsUnbinned = std::move(rhs.treeOutResidualsUnbinned);
  treeOutTrackData = std::move(rhs.treeOutTrackData);
  treeOutResiduals = std::move(rhs.treeOutResiduals);
  treeOutStats = std::move(rhs.treeOutStats);
  treeOutRecords = std::move(rhs.treeOutRecords);
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    residuals[iSec] = std::move(rhs.residuals[iSec]);
    stats[iSec] = std::move(rhs.stats[iSec]);
  }
  tfOrbits = std::move(rhs.tfOrbits);
  sumBinnedResid = std::move(rhs.sumBinnedResid);
  sumUnbinnedResid = std::move(rhs.sumUnbinnedResid);
  lumi = std::move(rhs.lumi);
  unbinnedRes = std::move(rhs.unbinnedRes);
  trackInfo = std::move(rhs.trackInfo);
  trkData = std::move(rhs.trkData);
  orbitReset = rhs.orbitReset;
  firstTForbit = rhs.firstTForbit;
  firstSeenTF = rhs.firstSeenTF;
  lastSeenTF = rhs.lastSeenTF;
  nResidualsTotal = rhs.nResidualsTotal;
}

void ResidualsContainer::init(const TrackResiduals* residualsEngine, std::string outputDir, bool wFile, bool wBinnedResid, bool wUnbinnedResid, bool wTrackData, int autosave, int compression)
{
  trackResiduals = residualsEngine;
  writeToRootFile = wFile;
  writeBinnedResid = wBinnedResid;
  writeUnbinnedResiduals = wUnbinnedResid;
  writeTrackData = wTrackData;
  autosaveInterval = autosave;
  if (writeToRootFile) {
    fileName += std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    fileName += ".root";
    std::string fileNameTmp = outputDir + fileName;
    fileNameTmp += ".part"; // to prevent premature external usage of the file use temporary name
    fileOut = std::make_unique<TFile>(fileNameTmp.c_str(), "recreate", "", compression);
  }
  if (writeUnbinnedResiduals) {
    treeOutResidualsUnbinned = std::make_unique<TTree>("unbinnedResid", "TPC unbinned residuals");
    treeOutResidualsUnbinned->Branch("res", &unbinnedResPtr);
    treeOutResidualsUnbinned->Branch("trackInfo", &trackInfoPtr);
    treeOutResidualsUnbinned->Branch("lumi", &lumiTF);
    treeOutResidualsUnbinned->Branch("timeMS", &timeMS);
  }
  if (writeTrackData) {
    treeOutTrackData = std::make_unique<TTree>("trackData", "Track information incl cluster range ref");
    treeOutTrackData->Branch("trk", &trkDataPtr);
  }
  if (writeBinnedResid) {
    treeOutResiduals = std::make_unique<TTree>("resid", "TPC binned residuals");
    treeOutStats = std::make_unique<TTree>("stats", "Voxel statistics mean position and nEntries");
    for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
      residualsPtr[iSec] = &residuals[iSec];
      statsPtr[iSec] = &stats[iSec];
      stats[iSec].resize(trackResiduals->getNVoxelsPerSector());
      for (int ix = 0; ix < trackResiduals->getNXBins(); ++ix) {
        for (int ip = 0; ip < trackResiduals->getNY2XBins(); ++ip) {
          for (int iz = 0; iz < trackResiduals->getNZ2XBins(); ++iz) {
            auto& statsVoxel = stats[iSec][trackResiduals->getGlbVoxBin(ix, ip, iz)];
            // COG estimates are set to the bin center by default
            trackResiduals->getVoxelCoordinates(iSec, ix, ip, iz, statsVoxel.meanPos[TrackResiduals::VoxX], statsVoxel.meanPos[TrackResiduals::VoxF], statsVoxel.meanPos[TrackResiduals::VoxZ]);
          }
        }
      }
      treeOutResiduals->Branch(Form("sec%d", iSec), &residualsPtr[iSec]);
      treeOutStats->Branch(Form("sec%d", iSec), &statsPtr[iSec]);
    }
  }
  treeOutRecords = std::make_unique<TTree>("records", "Statistics per TF slot");
  treeOutRecords->Branch("firstTForbit", &tfOrbitsPtr);
  treeOutRecords->Branch("sumOfBinnedResiduals", &sumBinnedResidPtr);
  treeOutRecords->Branch("sumOfUnbinnedResiduals", &sumUnbinnedResidPtr);
  treeOutRecords->Branch("lumi", &lumiPtr);
  LOG(debug) << "Done initializing residuals container for file named " << fileName;
}

void ResidualsContainer::fillStatisticsBranches()
{
  // only called when the slot is finalized, otherwise treeOutStats
  // remains empty and we keep the statistics in memory in the vectors
  // (since their size anyway does not change)
  treeOutRecords->Fill();
  if (writeBinnedResid) {
    treeOutStats->Fill();
  }
}

void ResidualsContainer::fill(const o2::dataformats::TFIDInfo& ti, const gsl::span<const UnbinnedResid> resid, const gsl::span<const o2::tpc::TrackDataCompact> trkRefsIn, long orbitResetTime, const gsl::span<const o2::tpc::TrackData>* trkDataIn, const o2::ctp::LumiInfo* lumiInput)
{
  // receives large vector of unbinned residuals and fills the sector-wise vectors
  // with binned residuals and statistics
  LOG(debug) << "Filling ResidualsContainer with vector of size " << resid.size();
  uint32_t nUnbinnedResidualsInTF = 0;
  uint32_t nBinnedResidualsInTF = 0;
  if (ti.tfCounter > lastSeenTF) {
    lastSeenTF = ti.tfCounter;
  }
  if (ti.tfCounter < firstSeenTF) {
    orbitReset = orbitResetTime;
    firstTForbit = ti.firstTForbit;
    firstSeenTF = ti.tfCounter;
  }
  for (const auto& residIn : resid) {
    ++nUnbinnedResidualsInTF;
    bool counterIncremented = false;
    if (writeUnbinnedResiduals) {
      unbinnedRes.push_back(residIn);
      ++nResidualsTotal;
      counterIncremented = true;
    }
    if (!writeBinnedResid) {
      continue;
    }
    int sec = residIn.sec;
    auto& residVecOut = residuals[sec];
    auto& statVecOut = stats[sec];
    std::array<unsigned char, TrackResiduals::VoxDim> bvox;
    float xPos = param::RowX[residIn.row];
    float yPos = residIn.y * param::MaxY / 0x7fff + residIn.dy * param::MaxResid / 0x7fff;
    float zPos = residIn.z * param::MaxZ / 0x7fff + residIn.dz * param::MaxResid / 0x7fff;
    if (!trackResiduals->findVoxelBin(sec, xPos, yPos, zPos, bvox)) {
      // we are not inside any voxel
      LOGF(debug, "Dropping residual in sec(%i), x(%f), y(%f), z(%f)", sec, xPos, yPos, zPos);
      continue;
    }
    residVecOut.emplace_back(residIn.dy, residIn.dz, residIn.tgSlp, bvox);
    auto& stat = statVecOut[trackResiduals->getGlbVoxBin(bvox)];
    float& binEntries = stat.nEntries;
    float oldEntries = binEntries++;
    float norm = 1.f / binEntries;
    // update COG for voxel bvox (update for X only needed in case binning is not per pad row)
    float xPosInv = 1.f / xPos;
    stat.meanPos[TrackResiduals::VoxX] = (stat.meanPos[TrackResiduals::VoxX] * oldEntries + xPos) * norm;
    stat.meanPos[TrackResiduals::VoxF] = (stat.meanPos[TrackResiduals::VoxF] * oldEntries + yPos * xPosInv) * norm;
    stat.meanPos[TrackResiduals::VoxZ] = (stat.meanPos[TrackResiduals::VoxZ] * oldEntries + zPos * xPosInv) * norm;
    if (!counterIncremented) {
      ++nResidualsTotal;
    }
    ++nBinnedResidualsInTF;
  }
  for (const auto& trkRef : trkRefsIn) {
    trackInfo.push_back(trkRef);
  }
  sumBinnedResid.push_back(nBinnedResidualsInTF);
  sumUnbinnedResid.push_back(nUnbinnedResidualsInTF);
  if (writeBinnedResid) {
    treeOutResiduals->Fill();
  }
  for (auto& residVecOut : residuals) {
    residVecOut.clear();
  }
  if (writeTrackData) {
    for (const auto& trkIn : *trkDataIn) {
      trkData.push_back(trkIn);
    }
    treeOutTrackData->Fill();
    trkData.clear();
  }
  if (writeUnbinnedResiduals) {
    if (lumiInput) {
      lumiTF = *lumiInput;
    }
    timeMS = orbitResetTime + ti.tfCounter * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
    treeOutResidualsUnbinned->Fill();
    unbinnedRes.clear();
    trackInfo.clear();
  }
  tfOrbits.push_back(ti.firstTForbit);
  if (lumiInput) {
    lumi.push_back(*lumiInput);
  }

  if (autosaveInterval > 0 && (tfOrbits.size() % autosaveInterval) == 0 && writeToRootFile) {
    writeToFile(false);
  }
}

void ResidualsContainer::writeToFile(bool closeFileAfterwards)
{
  LOG(info) << "Writing results to file. Closing afterwards? " << closeFileAfterwards;
  fillStatisticsBranches(); // these would need to be filled only once, so only the last entry is important
  fileOut->cd();
  treeOutRecords->Write();
  if (writeBinnedResid) {
    treeOutResiduals->Write();
    treeOutStats->Write();
  }
  if (writeUnbinnedResiduals) {
    treeOutResidualsUnbinned->Write();
  }
  if (writeTrackData) {
    treeOutTrackData->Write();
  }

  if (closeFileAfterwards) {
    treeOutRecords.reset();
    if (writeBinnedResid) {
      treeOutResiduals.reset();
      treeOutStats.reset();
    }
    if (writeUnbinnedResiduals) {
      treeOutResidualsUnbinned.reset();
    }
    if (writeTrackData) {
      treeOutTrackData.reset();
    }
    fileOut->Close();
    fileOut.reset();
  }
}

void ResidualsContainer::merge(ResidualsContainer* prev)
{
  // the previous slot is merged to this one and afterwards
  // the previous one will be deleted
  LOGP(debug, "Merging previous slot with {} entries into current one with {} entries", prev->getNEntries(), getNEntries());
  if (writeBinnedResid) {
    for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
      // merge statistics
      const auto& statVecPrev = prev->stats[iSec];
      auto& statVec = stats[iSec];
      for (int iVox = 0; iVox < trackResiduals->getNVoxelsPerSector(); ++iVox) {
        const auto& statPrev = statVecPrev[iVox];
        auto& stat = statVec[iVox];
        float norm = 1.f;
        if (statPrev.nEntries + stat.nEntries > 0.1) {
          // if there is at least a single entry in either of the containers we need the proper norm
          norm /= (statPrev.nEntries + stat.nEntries);
        }
        stat.meanPos[TrackResiduals::VoxX] = (stat.meanPos[TrackResiduals::VoxX] * stat.nEntries + statPrev.meanPos[TrackResiduals::VoxX] * statPrev.nEntries) * norm;
        stat.meanPos[TrackResiduals::VoxF] = (stat.meanPos[TrackResiduals::VoxF] * stat.nEntries + statPrev.meanPos[TrackResiduals::VoxF] * statPrev.nEntries) * norm;
        stat.meanPos[TrackResiduals::VoxZ] = (stat.meanPos[TrackResiduals::VoxZ] * stat.nEntries + statPrev.meanPos[TrackResiduals::VoxZ] * statPrev.nEntries) * norm;
        stat.nEntries += statPrev.nEntries;
      }
      // prepare merging of residuals
      prev->treeOutResiduals->SetBranchAddress(Form("sec%d", iSec), &residualsPtr[iSec]);
    }
    // We append the entries of the tree of the following slot to the
    // previous slot and afterwards move the merged tree to this slot.
    // This way the order of the entries is preserved
    for (int i = 0; i < treeOutResiduals->GetEntries(); ++i) {
      treeOutResiduals->GetEntry(i);
      prev->treeOutResiduals->Fill();
    }
  }

  if (writeTrackData) {
    prev->treeOutTrackData->SetBranchAddress("trk", &trkDataPtr);
    for (int i = 0; i < treeOutTrackData->GetEntries(); ++i) {
      treeOutTrackData->GetEntry(i);
      prev->treeOutTrackData->Fill();
    }
  }
  if (writeUnbinnedResiduals) {
    prev->treeOutResidualsUnbinned->SetBranchAddress("res", &unbinnedResPtr);
    prev->treeOutResidualsUnbinned->SetBranchAddress("trackInfo", &trackInfoPtr);
    for (int i = 0; i < treeOutResidualsUnbinned->GetEntries(); ++i) {
      treeOutResidualsUnbinned->GetEntry(i);
      prev->treeOutResidualsUnbinned->Fill();
    }
  }

  treeOutResiduals = std::move(prev->treeOutResiduals);
  treeOutTrackData = std::move(prev->treeOutTrackData);
  treeOutResidualsUnbinned = std::move(prev->treeOutResidualsUnbinned);

  // since we want to continue using the TTrees of the previous slot, we must
  // avoid that ROOT deletes them when the TFile of the previous slot is erased
  treeOutResiduals->SetDirectory(fileOut.get());
  treeOutTrackData->SetDirectory(fileOut.get());
  treeOutResidualsUnbinned->SetDirectory(fileOut.get());

  nResidualsTotal += prev->nResidualsTotal;

  // append the current vector to the vector of the previous container and afterwards swap them,
  // since the vector of the previous container will be deleted
  prev->tfOrbits.insert(prev->tfOrbits.end(), tfOrbits.begin(), tfOrbits.end());
  std::swap(prev->tfOrbits, tfOrbits);
  prev->sumBinnedResid.insert(prev->sumBinnedResid.end(), sumBinnedResid.begin(), sumBinnedResid.end());
  std::swap(prev->sumBinnedResid, sumBinnedResid);
  prev->sumUnbinnedResid.insert(prev->sumUnbinnedResid.end(), sumUnbinnedResid.begin(), sumUnbinnedResid.end());
  std::swap(prev->sumUnbinnedResid, sumUnbinnedResid);
  prev->lumi.insert(prev->lumi.end(), lumi.begin(), lumi.end());
  std::swap(prev->lumi, lumi);

  firstSeenTF = prev->firstSeenTF;
  LOGP(debug, "Done with the merge. Current slot has {} entries", getNEntries());
}

void ResidualsContainer::print()
{
  LOG(info) << "There are in total " << nResidualsTotal << " residuals stored in the container";
}

using Slot = o2::calibration::TimeSlot<ResidualsContainer>;

ResidualAggregator::~ResidualAggregator()
{
  auto& slots = getSlots();
  slots.clear();
}

bool ResidualAggregator::hasEnoughData(const Slot& slot) const
{
  LOG(debug) << "There are " << slot.getContainer()->getNEntries() << " entries currently. Min entries per voxel: " << mMinEntries;
  auto entriesPerVoxel = slot.getContainer()->getNEntries() / (mTrackResiduals.getNVoxelsPerSector() * SECTORSPERSIDE * SIDES);
  LOGP(debug, "Slot has {} entries per voxel, at least {} are required", entriesPerVoxel, mMinEntries);
  return entriesPerVoxel >= mMinEntries;
}

void ResidualAggregator::initOutput()
{
  // nothing to be done here, but method needs to be overwritten
}

void ResidualAggregator::finalizeSlot(Slot& slot)
{
  LOG(info) << "Finalizing slot";
  auto finalizeStartTime = std::chrono::high_resolution_clock::now();
  auto cont = slot.getContainer();
  cont->print();
  if (!mWriteOutput || cont->getNEntries() == 0) {
    LOGP(info, "Skip writing output with {} entries, since file output is disabled or slot is empty", cont->getNEntries());
    return;
  }
  cont->writeToFile(true);

  long orbitOffsetStart = (cont->firstSeenTF - slot.getTFStart()) * o2::base::GRPGeomHelper::getNHBFPerTF();
  long orbitOffsetEnd = (slot.getTFEnd() - cont->firstSeenTF) * o2::base::GRPGeomHelper::getNHBFPerTF();
  long timeStartMS = cont->orbitReset + (cont->firstTForbit - orbitOffsetStart) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
  long timeEndMS = cont->orbitReset + (cont->firstTForbit + orbitOffsetEnd) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
  auto fileName = fmt::format("o2tpc_residuals_{}_{}_{}_{}.root", timeStartMS, timeEndMS, slot.getTFStart(), slot.getTFEnd());
  auto fileNameWithPath = mOutputDir + fileName;
  std::filesystem::rename(o2::utils::Str::concat_string(mOutputDir, cont->fileName, ".part"), fileNameWithPath);
  if (mStoreMetaData) {
    o2::dataformats::FileMetaData fileMetaData; // object with information for meta data file
    fileMetaData.fillFileData(fileNameWithPath);
    fileMetaData.setDataTakingContext(mDataTakingContext);
    fileMetaData.type = "calib";
    fileMetaData.priority = "high";
    auto metaFileNameTmp = fmt::format("{}{}.tmp", mMetaOutputDir, fileName);
    auto metaFileName = fmt::format("{}{}.done", mMetaOutputDir, fileName);
    try {
      std::ofstream metaFileOut(metaFileNameTmp);
      metaFileOut << fileMetaData;
      metaFileOut.close();
      std::filesystem::rename(metaFileNameTmp, metaFileName);
    } catch (std::exception const& e) {
      LOG(error) << "Failed to store residuals meta data file " << metaFileName << ", reason: " << e.what();
    }
  }
  std::chrono::duration<double, std::milli> finalizeDuration = std::chrono::high_resolution_clock::now() - finalizeStartTime;
  LOGP(info, "Finalizing calibration slot took: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(finalizeDuration).count());
}

Slot& ResidualAggregator::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto emplaceStartTime = std::chrono::high_resolution_clock::now();
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tStart, tEnd) : cont.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<ResidualsContainer>());
  slot.getContainer()->init(&mTrackResiduals, mOutputDir, mWriteOutput, mWriteBinnedResiduals, mWriteUnbinnedResiduals, mWriteTrackData, mAutosaveInterval, mCompressionSetting);
  std::chrono::duration<double, std::milli> emplaceDuration = std::chrono::high_resolution_clock::now() - emplaceStartTime;
  LOGP(info, "Emplacing new calibration slot took: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(emplaceDuration).count());
  return slot;
}
