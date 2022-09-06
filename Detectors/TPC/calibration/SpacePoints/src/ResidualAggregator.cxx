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

using namespace o2::tpc;

ResidualsContainer::~ResidualsContainer()
{
  // trees must be deleted before the file is closed, otherwise segfaults
  treeOutResidualsUnbinned.reset();
  treeOutTrackData.reset();
  treeOutResiduals.reset();
  treeOutStats.reset();
  treeOutRecords.reset();
  if (fileOut) {
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
  trackResiduals = rhs.trackResiduals;
  fileOut = std::move(rhs.fileOut);
  fileName = std::move(rhs.fileName);
  treeOutResidualsUnbinned = std::move(rhs.treeOutResidualsUnbinned);
  treeOutTrackData = std::move(rhs.treeOutTrackData);
  treeOutResiduals = std::move(rhs.treeOutResiduals);
  treeOutStats = std::move(rhs.treeOutStats);
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    residuals[iSec] = std::move(rhs.residuals[iSec]);
    stats[iSec] = std::move(rhs.stats[iSec]);
  }
  runNumber = rhs.runNumber;
  tfOrbits = std::move(rhs.tfOrbits);
  sumOfResiduals = std::move(rhs.sumOfResiduals);
}

void ResidualsContainer::init(const TrackResiduals* residualsEngine, std::string outputDir, bool wFile, bool wBinnedResid, bool wUnbinnedResid, bool wTrackData, int autosave)
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
    fileOut = std::make_unique<TFile>(fileNameTmp.c_str(), "recreate");
  }
  if (writeUnbinnedResiduals) {
    treeOutResidualsUnbinned = std::make_unique<TTree>("unbinnedResid", "TPC unbinned residuals");
    treeOutResidualsUnbinned->Branch("res", &unbinnedResPtr);
  }
  if (writeTrackData) {
    treeOutTrackData = std::make_unique<TTree>("trackData", "Track information incl cluster range ref");
    treeOutTrackData->Branch("trk", &trkDataPtr);
  }
  if (writeBinnedResid) {
    treeOutResiduals = std::make_unique<TTree>(treeNameResiduals.c_str(), "TPC binned residuals");
    treeOutStats = std::make_unique<TTree>(treeNameStats.c_str(), "Voxel statistics mean position and nEntries");
    treeOutRecords = std::make_unique<TTree>(treeNameRecords.c_str(), "Statistics per TF slot");
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
    treeOutRecords->Branch("firstTForbit", &tfOrbitsPtr);
    treeOutRecords->Branch("sumOfResiduals", &sumOfResidualsPtr);
  }
}

void ResidualsContainer::fillStatisticsBranches()
{
  // only called when the slot is finalized, otherwise treeOutStats
  // remains empty and we keep the statistics in memory in the vectors
  // (since their size anyway does not change)
  if (writeBinnedResid) {
    treeOutStats->Fill();
    treeOutRecords->Fill();
  }
}

void ResidualsContainer::fill(const o2::dataformats::TFIDInfo& ti, const std::pair<gsl::span<const o2::tpc::TrackData>, gsl::span<const TrackResiduals::UnbinnedResid>> data)
{
  // receives large vector of unbinned residuals and fills the sector-wise vectors
  // with binned residuals and statistics
  LOG(debug) << "Filling ResidualsContainer with vector of size " << data.second.size();
  uint32_t nResidualsInTF = 0;
  for (const auto& residIn : data.second) {
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
    float yPos = residIn.y * param::MaxY / 0x7fff;
    float zPos = residIn.z * param::MaxZ / 0x7fff;
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
    ++nResidualsInTF;
  }
  if (writeBinnedResid) {
    treeOutResiduals->Fill();
    sumOfResiduals.push_back(nResidualsInTF);
  }
  for (auto& residVecOut : residuals) {
    residVecOut.clear();
  }
  if (writeTrackData) {
    for (const auto& trkIn : data.first) {
      trkData.push_back(trkIn);
    }
    treeOutTrackData->Fill();
    trkData.clear();
  }
  if (writeUnbinnedResiduals) {
    treeOutResidualsUnbinned->Fill();
    unbinnedRes.clear();
  }
  runNumber = ti.runNumber;
  tfOrbits.push_back(ti.firstTForbit);

  if (autosaveInterval > 0 && (tfOrbits.size() % autosaveInterval) == 0 && writeToRootFile) {
    writeToFile(false);
  }
}

void ResidualsContainer::writeToFile(bool closeFileAfterwards)
{
  LOG(info) << "Writing results to file. Closing afterwards? " << closeFileAfterwards;
  fillStatisticsBranches(); // these would need to be filled only once, so only the last entry is important
  fileOut->cd();
  if (writeBinnedResid) {
    treeOutResiduals->Write();
    treeOutStats->Write();
    treeOutRecords->Write();
  }
  if (writeUnbinnedResiduals) {
    treeOutResidualsUnbinned->Write();
  }
  if (writeTrackData) {
    treeOutTrackData->Write();
  }

  if (closeFileAfterwards) {
    if (writeBinnedResid) {
      treeOutResiduals.reset();
      treeOutStats.reset();
      treeOutRecords.reset();
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
  LOG(info) << "Merging previous slot into current one";
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
      prev->treeOutResiduals->SetBranchAddress(Form("sec%d", iSec), &residualsPtr);
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
    for (int i = 0; i < treeOutResidualsUnbinned->GetEntries(); ++i) {
      treeOutResidualsUnbinned->GetEntry(i);
      prev->treeOutResidualsUnbinned->Fill();
    }
  }

  treeOutResiduals = std::move(prev->treeOutResiduals);
  treeOutTrackData = std::move(prev->treeOutTrackData);
  treeOutResidualsUnbinned = std::move(prev->treeOutResidualsUnbinned);

  nResidualsTotal += prev->nResidualsTotal;

  // append the current vector to the vector of the previous container and afterwards swap them,
  // since the vector of the previous container will be deleted
  prev->tfOrbits.insert(prev->tfOrbits.end(), tfOrbits.begin(), tfOrbits.end());
  std::swap(prev->tfOrbits, tfOrbits);
  prev->sumOfResiduals.insert(prev->sumOfResiduals.end(), sumOfResiduals.begin(), sumOfResiduals.end());
  std::swap(prev->sumOfResiduals, sumOfResiduals);
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
  LOG(debug) << "There are " << slot.getContainer()->getNEntries() << " entries currently. Min entries prt voxel: " << mMinEntries;
  auto entriesPerVoxel = slot.getContainer()->getNEntries() / (mTrackResiduals.getNVoxelsPerSector() * SECTORSPERSIDE * SIDES);
  return entriesPerVoxel >= mMinEntries;
}

void ResidualAggregator::initOutput()
{
  // nothing to be done here, but method needs to be overwritten
}

void ResidualAggregator::finalizeSlot(Slot& slot)
{
  LOG(info) << "Finalizing slot";
  auto cont = slot.getContainer();
  cont->print();
  if (!mWriteOutput) {
    LOG(info) << "Skip writing output, since file output is disabled";
    return;
  }
  cont->writeToFile(true);
  std::filesystem::rename(o2::utils::Str::concat_string(mOutputDir, cont->fileName, ".part"), mOutputDir + cont->fileName);
  if (mStoreMetaData) {
    o2::dataformats::FileMetaData fileMetaData; // object with information for meta data file
    fileMetaData.fillFileData(mOutputDir + cont->fileName);
    fileMetaData.setDataTakingContext(mDataTakingContext);
    fileMetaData.type = "calib";
    fileMetaData.priority = "high";
    auto metaFileNameTmp = fmt::format("{}{}.tmp", mMetaOutputDir, cont->fileName);
    auto metaFileName = fmt::format("{}{}.done", mMetaOutputDir, cont->fileName);
    try {
      std::ofstream metaFileOut(metaFileNameTmp);
      metaFileOut << fileMetaData;
      metaFileOut << "TFOrbits: ";
      for (size_t i = 0; i < cont->tfOrbits.size(); i++) {
        metaFileOut << fmt::format("{}{}", i ? ", " : "", cont->tfOrbits[i]);
      }
      metaFileOut << '\n';
      metaFileOut.close();
      std::filesystem::rename(metaFileNameTmp, metaFileName);
    } catch (std::exception const& e) {
      LOG(error) << "Failed to store residuals meta data file " << metaFileName << ", reason: " << e.what();
    }
  }
}

Slot& ResidualAggregator::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tStart, tEnd) : cont.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<ResidualsContainer>());
  slot.getContainer()->init(&mTrackResiduals, mOutputDir, mWriteOutput, mWriteBinnedResiduals, mWriteUnbinnedResiduals, mWriteTrackData, mAutosaveInterval);
  return slot;
}
