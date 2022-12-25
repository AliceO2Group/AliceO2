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

/// @file   TPCResidualReaderSpec.cxx
/// \brief Reads binned residuals which are written by the aggregator and creates the static distortion maps for the TPC
/// \author Ole Schmidt

#include <memory>
#include <vector>
#include <boost/algorithm/string/predicate.hpp>
#include "TFile.h"
#include "TTree.h"
#include "TGrid.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DataFormatsTPC/Defs.h"
#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/TrackInterpolation.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/StringUtils.h"
#include "TPCInterpolationWorkflow/TPCResidualReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCResidualReader : public Task
{
 public:
  TPCResidualReader(bool doBinning, GID::mask_t src) : mDoBinning(doBinning), mSources(src) {}
  ~TPCResidualReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  /// fill residuals for one sector
  /// \param iSec sector of the residuals
  void fillResiduals(const int iSec);

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTreeResiduals;
  std::unique_ptr<TTree> mTreeStats;
  std::unique_ptr<TTree> mTreeUnbinnedResiduals;
  bool mDoBinning{false};
  bool mStoreBinnedResiduals{false};
  GID::mask_t mSources;
  TrackResiduals mTrackResiduals;
  std::vector<std::string> mFileNames;                                                              ///< input files
  std::string mOutfile{"debugVoxRes.root"};                                                         ///< output file name
  std::vector<TrackResiduals::LocalResid> mResiduals, *mResidualsPtr = &mResiduals;                 ///< binned residuals input
  std::array<std::vector<TrackResiduals::LocalResid>, SECTORSPERSIDE * SIDES> mResidualsSector;     ///< binned residuals generated on-the-fly
  std::array<std::vector<TrackResiduals::LocalResid>*, SECTORSPERSIDE * SIDES> mResidualsSectorPtr; ///< for setting branch addresses
  std::array<std::vector<TrackResiduals::VoxStats>, SECTORSPERSIDE * SIDES> mVoxStatsSector;        ///< voxel statistics generated on-the-fly
  std::vector<UnbinnedResid> mUnbinnedResiduals, *mUnbinnedResidualsPtr = &mUnbinnedResiduals;      ///< unbinned residuals input
  std::vector<TrackDataCompact> mTrackData, *mTrackDataPtr = &mTrackData;                           ///< the track references for unbinned residuals
};

void TPCResidualReader::fillResiduals(const int iSec)
{
  auto brStats = mTreeStats->GetBranch(Form("sec%d", iSec));
  brStats->SetAddress(mTrackResiduals.getVoxStatPtr());
  brStats->GetEntry(mTreeStats->GetEntries() - 1); // only the last entry is of interest
  mTrackResiduals.fillStats(iSec);

  // in case autosave was enabled, we have multiple entries in the statistics tree
  auto brResid = mTreeResiduals->GetBranch(Form("sec%d", iSec));
  brResid->SetAddress(&mResidualsPtr);

  for (int iEntry = 0; iEntry < brResid->GetEntries(); ++iEntry) {
    brResid->GetEntry(iEntry);
    LOGF(debug, "Pushing %lu TPC residuals at entry %i for sector %i", mResiduals.size(), iEntry, iSec);
    for (const auto& res : mResiduals) {
      LOGF(debug, "Adding residual from Voxel %i-%i-%i. dy(%i), dz(%i), tg(%i)", res.bvox[0], res.bvox[1], res.bvox[2], res.dy, res.dz, res.tgSlp);
    }
    mTrackResiduals.getLocalResVec().insert(mTrackResiduals.getLocalResVec().end(), mResiduals.begin(), mResiduals.end());
  }
}

void TPCResidualReader::init(InitContext& ic)
{
  const auto dontCheckFileAccess = ic.options().get<bool>("dont-check-file-access");

  auto fileList = o2::RangeTokenizer::tokenize<std::string>(ic.options().get<std::string>("residuals-infiles"));
  mOutfile = ic.options().get<std::string>("outfile");
  mTrackResiduals.init();

  // check if only one input file (a txt file contaning a list of files is provided)
  if (fileList.size() == 1) {
    if (boost::algorithm::ends_with(fileList.front(), "txt")) {
      LOGP(info, "Reading files from input file {}", fileList.front());
      std::ifstream is(fileList.front());
      std::istream_iterator<std::string> start(is);
      std::istream_iterator<std::string> end;
      std::vector<std::string> fileNamesTmp(start, end);
      fileList = fileNamesTmp;
    }
  }

  const std::string inpDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir"));
  for (const auto& file : fileList) {
    if ((file.find("alien://") == 0) && !gGrid && !TGrid::Connect("alien://")) {
      LOG(fatal) << "Failed to open alien connection";
    }
    const auto fileDir = o2::utils::Str::concat_string(inpDir, file);
    if (!dontCheckFileAccess) {
      std::unique_ptr<TFile> filePtr(TFile::Open(fileDir.data()));
      if (!filePtr || !filePtr->IsOpen() || filePtr->IsZombie()) {
        LOGP(warning, "Could not open file {}", fileDir);
        continue;
      }
    }
    mFileNames.emplace_back(fileDir);
  }

  if (mFileNames.size() == 0) {
    LOGP(error, "No input files to process");
  }

  mStoreBinnedResiduals = ic.options().get<bool>("store-binned");
}

void TPCResidualReader::run(ProcessingContext& pc)
{
  mTrackResiduals.createOutputFile(mOutfile.data()); // FIXME remove when map output is handled properly

  if (mDoBinning) {
    // initialize the trees for the binned residuals and the statistics
    LOGP(info, "Preparing the binning of residuals. Storing the afterwards is set to {}", mStoreBinnedResiduals);
    mTreeResiduals = std::make_unique<TTree>("resid", "TPC binned residuals");
    if (!mStoreBinnedResiduals) {
      // if not set to nullptr, the reisduals tree will be added to the output file of TrackResiduals created above
      mTreeResiduals->SetDirectory(nullptr);
    }
    for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
      mResidualsSectorPtr[iSec] = &mResidualsSector[iSec];
      mVoxStatsSector[iSec].resize(mTrackResiduals.getNVoxelsPerSector());
      for (int ix = 0; ix < mTrackResiduals.getNXBins(); ++ix) {
        for (int ip = 0; ip < mTrackResiduals.getNY2XBins(); ++ip) {
          for (int iz = 0; iz < mTrackResiduals.getNZ2XBins(); ++iz) {
            auto& statsVoxel = mVoxStatsSector[iSec][mTrackResiduals.getGlbVoxBin(ix, ip, iz)];
            // COG estimates are set to the bin center by default
            mTrackResiduals.getVoxelCoordinates(iSec, ix, ip, iz, statsVoxel.meanPos[TrackResiduals::VoxX], statsVoxel.meanPos[TrackResiduals::VoxF], statsVoxel.meanPos[TrackResiduals::VoxZ]);
          }
        }
      }
      mTreeResiduals->Branch(Form("sec%d", iSec), &mResidualsSectorPtr[iSec]);
    }
    // now go through all input files, apply the track selection and fill the binned residuals
    for (const auto& file : mFileNames) {
      LOGP(info, "Processing residuals from file {}", file);
      connectTree(file);
      for (int iEntry = 0; iEntry < mTreeUnbinnedResiduals->GetEntries(); ++iEntry) {
        mTreeUnbinnedResiduals->GetEntry(iEntry);
        for (const auto& trkInfo : mTrackData) {
          if (!GID::includesSource(trkInfo.sourceId, mSources)) {
            continue;
          }
          for (int i = trkInfo.idxFirstResidual; i < trkInfo.idxFirstResidual + trkInfo.nResiduals; ++i) {
            const auto& residIn = mUnbinnedResiduals[i];
            int sec = residIn.sec;
            auto& residVecOut = mResidualsSector[sec];
            auto& statVecOut = mVoxStatsSector[sec];
            std::array<unsigned char, TrackResiduals::VoxDim> bvox;
            float xPos = param::RowX[residIn.row];
            float yPos = residIn.y * param::MaxY / 0x7fff;
            float zPos = residIn.z * param::MaxZ / 0x7fff;
            if (!mTrackResiduals.findVoxelBin(sec, xPos, yPos, zPos, bvox)) {
              // we are not inside any voxel
              LOGF(debug, "Dropping residual in sec(%i), x(%f), y(%f), z(%f)", sec, xPos, yPos, zPos);
              continue;
            }
            residVecOut.emplace_back(residIn.dy, residIn.dz, residIn.tgSlp, bvox);
            auto& stat = statVecOut[mTrackResiduals.getGlbVoxBin(bvox)];
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
      mTreeResiduals->Fill();
      for (auto& residuals : mResidualsSector) {
        residuals.clear();
      }
    }
  }

  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    if (mDoBinning) {
      // for each sector fill the vector of local residuals from the respective branch
      auto brResid = mTreeResiduals->GetBranch(Form("sec%d", iSec));
      brResid->SetAddress(&mResidualsPtr);
      for (int iEntry = 0; iEntry < brResid->GetEntries(); ++iEntry) {
        brResid->GetEntry(iEntry);
        mTrackResiduals.getLocalResVec().insert(mTrackResiduals.getLocalResVec().end(), mResiduals.begin(), mResiduals.end());
      }
      mTrackResiduals.setStats(mVoxStatsSector[iSec], iSec);
    } else {
      // we read the binned residuals directly from the input files
      for (const auto& file : mFileNames) {
        LOGP(info, "Processing residuals from file {}", file);
        // set up the tree from the input file
        connectTree(file);
        // fill the residuals for one sector
        fillResiduals(iSec);
      }
    }

    // do processing
    mTrackResiduals.processSectorResiduals(iSec);
    // do cleanup
    mTrackResiduals.clear();
  }

  mTrackResiduals.closeOutputFile(); // FIXME remove when map output is handled properly

  // const auto& voxResArray = mTrackResiduals.getVoxelResults(); // array with one vector of results per sector
  // pc.outputs().snapshot(Output{"GLO", "VOXELRESULTS", 0, Lifetime::Timeframe}, voxResArray); // send results as one large vector?

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

void TPCResidualReader::connectTree(const std::string& filename)
{
  if (!mDoBinning) {
    // in case we do the binning on-the-fly, we fill these trees manually
    // and don't want to delete them in between
    mTreeResiduals.reset(nullptr); // in case it was already loaded
    mTreeStats.reset(nullptr);     // in case it was already loaded
  }
  mTreeUnbinnedResiduals.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  if (!mDoBinning) {
    // we load the already binned residuals and the voxel statistics
    mTreeResiduals.reset((TTree*)mFile->Get("resid"));
    assert(mTreeResiduals);
    mTreeStats.reset((TTree*)mFile->Get("stats"));
    assert(mTreeStats);
    LOG(info) << "Loaded tree from " << filename << " with " << mTreeResiduals->GetEntries() << " entries";
  } else {
    // we load the unbinned residuals
    LOG(info) << "Loading the binned residuals";
    mTreeUnbinnedResiduals.reset((TTree*)mFile->Get("unbinnedResid"));
    assert(mTreeUnbinnedResiduals);
    mTreeUnbinnedResiduals->SetBranchAddress("res", &mUnbinnedResidualsPtr);
    mTreeUnbinnedResiduals->SetBranchAddress("trackInfo", &mTrackDataPtr);
    LOG(info) << "Loaded tree from " << filename << " with " << mTreeUnbinnedResiduals->GetEntries() << " entries";
  }
}

DataProcessorSpec getTPCResidualReaderSpec(bool doBinning, GID::mask_t src)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  // outputs.emplace_back("GLO", "VOXELRESULTS", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "tpc-residual-reader",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCResidualReader>(doBinning, src)},
    Options{
      {"residuals-infiles", VariantType::String, "o2tpc_residuals.root", {"comma-separated list of input files or .txt file containing list of input files"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"outfile", VariantType::String, "debugVoxRes.root", {"Output file name"}},
      {"store-binned", VariantType::Bool, false, {"Store the binned residuals together with the voxel results"}},
      {"dont-check-file-access", VariantType::Bool, false, {"Deactivate check if all files are accessible before adding them to the list of files"}},
    }};
}

} // namespace tpc
} // namespace o2
