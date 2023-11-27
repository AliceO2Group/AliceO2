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

/// \file TPCTimeSeriesSpec.cxx
/// \brief device for time series
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Aug 20, 2023

#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/Mapper.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCWorkflow/TPCTimeSeriesSpec.h"
#include "DetectorsBase/TFIDInfoHelper.h"
#include "DetectorsBase/Propagator.h"
#include "TPCCalibration/RobustAverage.h"
#include "DetectorsCalibration/IntegratedClusterCalibrator.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "MathUtils/Tsallis.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "CommonDataFormat/AbstractRefAccessor.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCCalibration/VDriftHelper.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include <random>
#include <chrono>
#include "DataFormatsTPC/PIDResponse.h"
#include "DataFormatsITS/TrackITS.h"
#include "TROOT.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"

using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;
using TrkSrc = o2::dataformats::VtxTrackIndex::Source;
using DetID = o2::detectors::DetID;
using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCTimeSeries : public Task
{
 public:
  /// \constructor
  TPCTimeSeries(std::shared_ptr<o2::base::GRPGeomRequest> req, const bool disableWriter, const o2::base::Propagator::MatCorrType matType, const bool enableUnbinnedWriter, const bool tpcOnly, std::shared_ptr<o2::globaltracking::DataRequest> dr) : mCCDBRequest(req), mDisableWriter(disableWriter), mMatType(matType), mUnbinnedWriter(enableUnbinnedWriter), mTPCOnly(tpcOnly), mDataRequest(dr){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNMaxTracks = ic.options().get<int>("max-tracks");
    mMinMom = ic.options().get<float>("min-momentum");
    mMinNCl = ic.options().get<int>("min-cluster");
    mMaxTgl = ic.options().get<float>("max-tgl");
    mMaxQPt = ic.options().get<float>("max-qPt");
    mCoarseStep = ic.options().get<float>("coarse-step");
    mFineStep = ic.options().get<float>("fine-step");
    mCutDCA = ic.options().get<float>("cut-DCA-median");
    mCutRMS = ic.options().get<float>("cut-DCA-RMS");
    mRefXSec = ic.options().get<float>("refX-for-sector");
    mTglBins = ic.options().get<int>("tgl-bins");
    mPhiBins = ic.options().get<int>("phi-bins");
    mQPtBins = ic.options().get<int>("qPt-bins");
    mNThreads = ic.options().get<int>("threads");
    maxITSTPCDCAr = ic.options().get<float>("max-ITS-TPC-DCAr");
    maxITSTPCDCAz = ic.options().get<float>("max-ITS-TPC-DCAz");
    maxITSTPCDCAr_comb = ic.options().get<float>("max-ITS-TPC-DCAr_comb");
    maxITSTPCDCAz_comb = ic.options().get<float>("max-ITS-TPC-DCAz_comb");
    mTimeWindowMUS = ic.options().get<float>("time-window-mult-mus");
    mMIPdEdx = ic.options().get<float>("MIP-dedx");
    mMaxSnp = ic.options().get<float>("max-snp");
    mXCoarse = ic.options().get<float>("mX-coarse");
    mSqrt = ic.options().get<float>("sqrts");
    mMultBins = ic.options().get<int>("mult-bins");
    mMultMax = ic.options().get<int>("mult-max");
    mMinTracksPerVertex = ic.options().get<int>("min-tracks-per-vertex");
    mMaxdEdxRatio = ic.options().get<float>("max-dedx-ratio");
    mMaxdEdxRegionRatio = ic.options().get<float>("max-dedx-region-ratio");
    mSamplingFactor = ic.options().get<float>("sampling-factor");
    mSampleTsallis = ic.options().get<bool>("sample-unbinned-tsallis");
    if (mSampleTsallis) {
      mGenerator = std::mt19937(std::random_device{}());
    }
    mBufferVals.resize(mNThreads);
    mBufferDCA.setBinning(mPhiBins, mTglBins, mQPtBins, mMultBins, mMaxTgl, mMaxQPt, mMultMax);
    if (mUnbinnedWriter) {
      std::string outfile = ic.options().get<std::string>("out-file-unbinned");
      if (mNThreads > 1) {
        ROOT::EnableThreadSafety();
      }
      mStreamer.resize(mNThreads);
      for (int iThread = 0; iThread < mNThreads; ++iThread) {
        std::string outfileThr = outfile;
        outfileThr.replace(outfileThr.length() - 5, outfileThr.length(), fmt::format("_{}.root", iThread));
        LOGP(info, "Writing unbinned data to: {}", outfileThr);
        mStreamer[iThread] = std::make_unique<o2::utils::TreeStreamRedirector>(outfileThr.data(), "recreate");
      }
    }
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    mTPCVDriftHelper.extractCCDBInputs(pc);
    if (mTPCVDriftHelper.isUpdated()) {
      mTPCVDriftHelper.acknowledgeUpdate();
      mVDrift = mTPCVDriftHelper.getVDriftObject().getVDrift();
      LOGP(info, "Updated reference drift velocity to: {}", mVDrift);
    }

    const int nBins = getNBins();

    // init only once
    if (mAvgADCAr.size() != nBins) {
      mBufferDCA.resize(nBins);
      mAvgADCAr.resize(nBins);
      mAvgCDCAr.resize(nBins);
      mAvgADCAz.resize(nBins);
      mAvgCDCAz.resize(nBins);
      mAvgMeffA.resize(nBins);
      mAvgMeffC.resize(nBins);
      mAvgChi2MatchA.resize(nBins);
      mAvgChi2MatchC.resize(nBins);
      mMIPdEdxRatioQMaxA.resize(nBins);
      mMIPdEdxRatioQMaxC.resize(nBins);
      mMIPdEdxRatioQTotA.resize(nBins);
      mMIPdEdxRatioQTotC.resize(nBins);
      mTPCChi2A.resize(nBins);
      mTPCChi2C.resize(nBins);
      mTPCNClA.resize(nBins);
      mTPCNClC.resize(nBins);
      mLogdEdxQTotA.resize(nBins);
      mLogdEdxQTotC.resize(nBins);
      mLogdEdxQMaxA.resize(nBins);
      mLogdEdxQMaxC.resize(nBins);
      mITSPropertiesA.resize(nBins);
      mITSPropertiesC.resize(nBins);
      mITSTPCDeltaPA.resize(nBins);
      mITSTPCDeltaPC.resize(nBins);
      mSigmaYZA.resize(nBins);
      mSigmaYZC.resize(nBins);
    }

    RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest.get());

    // getting tracks
    auto tracksTPC = recoData.getTPCTracks();
    auto tracksITSTPC = mTPCOnly ? gsl::span<o2::dataformats::TrackTPCITS>() : recoData.getTPCITSTracks();
    auto tracksITS = mTPCOnly ? gsl::span<o2::its::TrackITS>() : recoData.getITSTracks();

    // getting the vertices
    auto vertices = mTPCOnly ? gsl::span<o2::dataformats::PrimaryVertex>() : recoData.getPrimaryVertices();
    auto primMatchedTracks = mTPCOnly ? gsl::span<o2::dataformats::VtxTrackIndex>() : recoData.getPrimaryVertexMatchedTracks();     // Global ID's for associated tracks
    auto primMatchedTracksRef = mTPCOnly ? gsl::span<o2::dataformats::VtxTrackRef>() : recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs

    // TOF clusters
    const auto& tofClusters = mTPCOnly ? gsl::span<o2::tof::Cluster>() : recoData.getTOFClusters();

    LOGP(info, "Processing {} vertices, {} primary matched vertices, {} TPC tracks, {} ITS tracks, {} ITS-TPC tracks, {} TOF clusters", vertices.size(), primMatchedTracks.size(), tracksTPC.size(), tracksITS.size(), tracksITSTPC.size(), tofClusters.size());

    // calculate mean vertex, RMS and count vertices
    auto indicesITSTPC_vtx = processVertices(vertices, primMatchedTracks, primMatchedTracksRef, recoData);

    // storing indices to ITS-TPC tracks and vertex ID for tpc track
    std::unordered_map<unsigned int, std::array<int, 2>> indicesITSTPC; // TPC track index -> ITS-TPC track index, vertex ID
    // loop over all ITS-TPC tracks
    for (int i = 0; i < tracksITSTPC.size(); ++i) {
      auto it = indicesITSTPC_vtx.find(i);
      // check if ITS-TPC track has attached vertex
      const auto idxVtx = (it != indicesITSTPC_vtx.end()) ? (it->second) : -1;
      // store TPC index and ITS-TPC+vertex index
      indicesITSTPC[tracksITSTPC[i].getRefTPC().getIndex()] = {i, idxVtx};
    }

    std::vector<std::tuple<int, float, float>> idxTPCTrackToTOFCluster; // store for each tpc track index the index to the TOF cluster

    // get matches to TOF in case skimmed data is produced
    if (mUnbinnedWriter) {
      idxTPCTrackToTOFCluster = std::vector<std::tuple<int, float, float>>(tracksTPC.size(), {-1, -999, -999});
      const std::vector<gsl::span<const o2::dataformats::MatchInfoTOF>> tofMatches{recoData.getTPCTOFMatches(), recoData.getTPCTRDTOFMatches(), recoData.getITSTPCTOFMatches(), recoData.getITSTPCTRDTOFMatches()};

      // loop over ITS-TPC-TRD-TOF and ITS-TPC-TOF tracks an store for each ITS-TPC track the TOF track index
      for (const auto& tofMatch : tofMatches) {
        for (const auto& tpctofmatch : tofMatch) {
          auto refTPC = recoData.getTPCContributorGID(tpctofmatch.getTrackRef());
          if (refTPC.isIndexSet()) {
            idxTPCTrackToTOFCluster[refTPC] = {tpctofmatch.getIdxTOFCl(), tpctofmatch.getDXatTOF(), tpctofmatch.getDZatTOF()};
          }
        }
      }
    }

    // find nearest vertex of tracks which have no vertex assigned
    findNearesVertex(tracksTPC, vertices);

    // getting cluster references for cluster bitmask
    if (mUnbinnedWriter) {
      mTPCTrackClIdx = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
      mFirstTFOrbit = processing_helpers::getFirstTForbit(pc);
    }

    // get local multiplicity - count neighbouring tracks
    findNNeighbourTracks(tracksTPC);

    // reset buffers
    for (int i = 0; i < nBins; ++i) {
      for (int type = 0; type < mAvgADCAr[i].size(); ++type) {
        mAvgADCAr[i][type].clear();
        mAvgCDCAr[i][type].clear();
        mAvgADCAz[i][type].clear();
        mAvgCDCAz[i][type].clear();
      }
      for (int type = 0; type < mMIPdEdxRatioQMaxA[i].size(); ++type) {
        mMIPdEdxRatioQMaxA[i][type].clear();
        mMIPdEdxRatioQMaxC[i][type].clear();
        mMIPdEdxRatioQTotA[i][type].clear();
        mMIPdEdxRatioQTotC[i][type].clear();
        mTPCChi2A[i][type].clear();
        mTPCChi2C[i][type].clear();
        mTPCNClA[i][type].clear();
        mTPCNClC[i][type].clear();
        mMIPdEdxRatioQMaxA[i][type].setUseWeights(false);
        mMIPdEdxRatioQMaxC[i][type].setUseWeights(false);
        mMIPdEdxRatioQTotA[i][type].setUseWeights(false);
        mMIPdEdxRatioQTotC[i][type].setUseWeights(false);
        mTPCChi2A[i][type].setUseWeights(false);
        mTPCChi2C[i][type].setUseWeights(false);
        mTPCNClA[i][type].setUseWeights(false);
        mTPCNClC[i][type].setUseWeights(false);
      }
      for (int type = 0; type < mLogdEdxQTotA[i].size(); ++type) {
        mLogdEdxQTotA[i][type].clear();
        mLogdEdxQTotC[i][type].clear();
        mLogdEdxQMaxA[i][type].clear();
        mLogdEdxQMaxC[i][type].clear();
        mLogdEdxQTotA[i][type].setUseWeights(false);
        mLogdEdxQTotC[i][type].setUseWeights(false);
        mLogdEdxQMaxA[i][type].setUseWeights(false);
        mLogdEdxQMaxC[i][type].setUseWeights(false);
      }
      for (int j = 0; j < mITSPropertiesA[i].size(); ++j) {
        mITSPropertiesA[i][j].clear();
        mITSPropertiesC[i][j].clear();
        mITSPropertiesA[i][j].setUseWeights(false);
        mITSPropertiesC[i][j].setUseWeights(false);
      }

      for (int j = 0; j < mITSTPCDeltaPA[i].size(); ++j) {
        mITSTPCDeltaPA[i][j].clear();
        mITSTPCDeltaPC[i][j].clear();
        mITSTPCDeltaPA[i][j].setUseWeights(false);
        mITSTPCDeltaPC[i][j].setUseWeights(false);
      }

      for (int j = 0; j < mSigmaYZA[i].size(); ++j) {
        mSigmaYZA[i][j].clear();
        mSigmaYZC[i][j].clear();
        mSigmaYZA[i][j].setUseWeights(false);
        mSigmaYZC[i][j].setUseWeights(false);
      }

      for (int j = 0; j < mAvgMeffA[i].size(); ++j) {
        mAvgMeffA[i][j].clear();
        mAvgMeffC[i][j].clear();
        mAvgChi2MatchA[i][j].clear();
        mAvgChi2MatchC[i][j].clear();
        mAvgMeffA[i][j].setUseWeights(false);
        mAvgMeffC[i][j].setUseWeights(false);
        mAvgChi2MatchA[i][j].setUseWeights(false);
        mAvgChi2MatchC[i][j].setUseWeights(false);
      }
    }

    for (int i = 0; i < mNThreads; ++i) {
      mBufferVals[i].front().clear();
      mBufferVals[i].back().clear();
    }

    // define number of tracks which are used
    const auto nTracks = tracksTPC.size();
    const size_t loopEnd = (mNMaxTracks < 0) ? nTracks : ((mNMaxTracks > nTracks) ? nTracks : size_t(mNMaxTracks));

    // reserve memory
    for (int i = 0; i < nBins; ++i) {
      const int lastIdxPhi = mBufferDCA.mTSTPC.getIndexPhi(mPhiBins);
      const int lastIdxTgl = mBufferDCA.mTSTPC.getIndexTgl(mTglBins);
      const int lastIdxQPt = mBufferDCA.mTSTPC.getIndexqPt(mQPtBins);
      const int lastIdxMult = mBufferDCA.mTSTPC.getIndexMult(mMultBins);
      const int firstIdx = mBufferDCA.mTSTPC.getIndexInt();
      int resMem = 0;
      if (i < lastIdxPhi) {
        resMem = loopEnd / mPhiBins;
      } else if (i < lastIdxTgl) {
        resMem = loopEnd / mTglBins;
      } else if (i < lastIdxQPt) {
        resMem = loopEnd / mQPtBins;
      } else if (i < lastIdxMult) {
        resMem = loopEnd / mMultBins;
      } else {
        resMem = loopEnd;
      }
      // Divide by 2 for A-C-side
      resMem /= 2;
      for (int type = 0; type < mAvgADCAr[i].size(); ++type) {
        mAvgADCAr[i][type].reserve(resMem);
        mAvgCDCAr[i][type].reserve(resMem);
        mAvgADCAz[i][type].reserve(resMem);
        mAvgCDCAz[i][type].reserve(resMem);
      }
      for (int type = 0; type < mMIPdEdxRatioQMaxA[i].size(); ++type) {
        mMIPdEdxRatioQMaxA[i][type].reserve(resMem);
        mMIPdEdxRatioQMaxC[i][type].reserve(resMem);
        mMIPdEdxRatioQTotA[i][type].reserve(resMem);
        mMIPdEdxRatioQTotC[i][type].reserve(resMem);
        mTPCChi2A[i][type].reserve(resMem);
        mTPCChi2C[i][type].reserve(resMem);
        mTPCNClA[i][type].reserve(resMem);
        mTPCNClC[i][type].reserve(resMem);
      }
      for (int j = 0; j < mAvgMeffA[i].size(); ++j) {
        mAvgMeffA[i][j].reserve(resMem);
        mAvgMeffC[i][j].reserve(resMem);
        mAvgChi2MatchA[i][j].reserve(resMem);
        mAvgChi2MatchC[i][j].reserve(resMem);
      }
      for (int j = 0; j < mITSPropertiesA[i].size(); ++j) {
        mITSPropertiesA[i][j].reserve(resMem);
        mITSPropertiesC[i][j].reserve(resMem);
      }
      for (int j = 0; j < mITSTPCDeltaPA[i].size(); ++j) {
        mITSTPCDeltaPA[i][j].reserve(resMem);
        mITSTPCDeltaPC[i][j].reserve(resMem);
      }
      for (int j = 0; j < mSigmaYZA[i].size(); ++j) {
        mSigmaYZA[i][j].reserve(resMem);
        mSigmaYZC[i][j].reserve(resMem);
      }
      for (int j = 0; j < mAvgMeffA[i].size(); ++j) {
        mLogdEdxQTotA[i][j].reserve(resMem);
        mLogdEdxQTotC[i][j].reserve(resMem);
        mLogdEdxQMaxA[i][j].reserve(resMem);
        mLogdEdxQMaxC[i][j].reserve(resMem);
      }
    }
    for (int iThread = 0; iThread < mNThreads; ++iThread) {
      const int resMem = (mNThreads > 0) ? loopEnd / mNThreads : loopEnd;
      mBufferVals[iThread].front().reserve(loopEnd, 1);
      mBufferVals[iThread].back().reserve(loopEnd, 0);
    }

    using timer = std::chrono::high_resolution_clock;
    auto startTotal = timer::now();

    // loop over tracks and calculate DCAs
    if (loopEnd < nTracks) {
      // draw random tracks
      std::vector<size_t> ind(nTracks);
      std::iota(ind.begin(), ind.end(), 0);
      std::minstd_rand rng(std::time(nullptr));
      std::shuffle(ind.begin(), ind.end(), rng);

      auto myThread = [&](int iThread) {
        for (size_t i = iThread; i < loopEnd; i += mNThreads) {
          if (acceptTrack(tracksTPC[i])) {
            fillDCA(tracksTPC, tracksITSTPC, vertices, i, iThread, indicesITSTPC, tracksITS, idxTPCTrackToTOFCluster, tofClusters);
          }
        }
      };

      std::vector<std::thread> threads(mNThreads);
      for (int i = 0; i < mNThreads; i++) {
        threads[i] = std::thread(myThread, i);
      }

      for (auto& th : threads) {
        th.join();
      }
    } else {
      auto myThread = [&](int iThread) {
        for (size_t i = iThread; i < loopEnd; i += mNThreads) {
          if (acceptTrack(tracksTPC[i])) {
            fillDCA(tracksTPC, tracksITSTPC, vertices, i, iThread, indicesITSTPC, tracksITS, idxTPCTrackToTOFCluster, tofClusters);
          }
        }
      };

      std::vector<std::thread> threads(mNThreads);
      for (int i = 0; i < mNThreads; i++) {
        threads[i] = std::thread(myThread, i);
      }

      for (auto& th : threads) {
        th.join();
      }
    }

    // fill DCA values from buffer
    for (const auto& vals : mBufferVals) {
      for (int type = 0; type < vals.size(); ++type) {
        const auto& val = vals[type];
        const auto nPoints = val.side.size();
        for (int i = 0; i < nPoints; ++i) {
          const auto tglBin = val.tglBin[i];
          const auto phiBin = val.phiBin[i];
          const auto qPtBin = val.qPtBin[i];
          const auto multBin = val.multBin[i];
          const auto dcar = val.dcar[i];
          const auto dcaz = val.dcaz[i];
          const auto dcarW = val.dcarW[i];
          const int binInt = nBins - 1;
          const bool fillCombDCA = ((type == 1) && (val.dcarcomb[i] != -1) && (val.dcazcomb[i] != -1));
          const std::array<int, 5> bins{tglBin, phiBin, qPtBin, multBin, binInt};
          // fill bins
          for (auto bin : bins) {
            if (val.side[i] == Side::C) {
              mAvgCDCAr[bin][type].addValue(dcar, dcarW);
              if (fillCombDCA) {
                mAvgCDCAr[bin][2].addValue(val.dcarcomb[i], dcarW);
                mAvgCDCAz[bin][2].addValue(val.dcazcomb[i], dcarW);
              }
              // fill only in case of valid value
              if (dcaz != 0) {
                mAvgCDCAz[bin][type].addValue(dcaz, dcarW);
              }
            } else {
              mAvgADCAr[bin][type].addValue(dcar, dcarW);
              if (fillCombDCA) {
                mAvgADCAr[bin][2].addValue(val.dcarcomb[i], dcarW);
                mAvgADCAz[bin][2].addValue(val.dcazcomb[i], dcarW);
              }
              // fill only in case of valid value
              if (dcaz != 0) {
                mAvgADCAz[bin][type].addValue(dcaz, dcarW);
              }
            }
          }
        }
      }
    }

    // calculate statistics and store values
    // loop over TPC sides
    for (int type = 0; type < 2; ++type) {
      // loop over phi and tgl bins
      for (int slice = 0; slice < nBins; ++slice) {
        auto& bufferDCA = (type == 0) ? mBufferDCA.mTSTPC : mBufferDCA.mTSITSTPC;

        const auto dcaAr = mAvgADCAr[slice][type].filterPointsMedian(mCutDCA, mCutRMS);
        bufferDCA.mDCAr_A_Median[slice] = std::get<0>(dcaAr);
        bufferDCA.mDCAr_A_WeightedMean[slice] = std::get<1>(dcaAr);
        bufferDCA.mDCAr_A_RMS[slice] = std::get<2>(dcaAr);
        bufferDCA.mDCAr_A_NTracks[slice] = std::get<3>(dcaAr);

        const auto dcaAz = mAvgADCAz[slice][type].filterPointsMedian(mCutDCA, mCutRMS);
        bufferDCA.mDCAz_A_Median[slice] = std::get<0>(dcaAz);
        bufferDCA.mDCAz_A_WeightedMean[slice] = std::get<1>(dcaAz);
        bufferDCA.mDCAz_A_RMS[slice] = std::get<2>(dcaAz);
        bufferDCA.mDCAz_A_NTracks[slice] = std::get<3>(dcaAz);

        const auto dcaCr = mAvgCDCAr[slice][type].filterPointsMedian(mCutDCA, mCutRMS);
        bufferDCA.mDCAr_C_Median[slice] = std::get<0>(dcaCr);
        bufferDCA.mDCAr_C_WeightedMean[slice] = std::get<1>(dcaCr);
        bufferDCA.mDCAr_C_RMS[slice] = std::get<2>(dcaCr);
        bufferDCA.mDCAr_C_NTracks[slice] = std::get<3>(dcaCr);

        const auto dcaCz = mAvgCDCAz[slice][type].filterPointsMedian(mCutDCA, mCutRMS);
        bufferDCA.mDCAz_C_Median[slice] = std::get<0>(dcaCz);
        bufferDCA.mDCAz_C_WeightedMean[slice] = std::get<1>(dcaCz);
        bufferDCA.mDCAz_C_RMS[slice] = std::get<2>(dcaCz);
        bufferDCA.mDCAz_C_NTracks[slice] = std::get<3>(dcaCz);
        // store combined ITS-TPC DCAs
        if (type == 1) {
          const auto dcaArComb = mAvgADCAr[slice][2].filterPointsMedian(mCutDCA, mCutRMS);
          mBufferDCA.mDCAr_comb_A_Median[slice] = std::get<0>(dcaArComb);
          mBufferDCA.mDCAr_comb_A_RMS[slice] = std::get<2>(dcaArComb);

          const auto dcaAzCom = mAvgADCAz[slice][2].filterPointsMedian(mCutDCA, mCutRMS);
          mBufferDCA.mDCAz_comb_A_Median[slice] = std::get<0>(dcaAzCom);
          mBufferDCA.mDCAz_comb_A_RMS[slice] = std::get<2>(dcaAzCom);

          const auto dcaCrComb = mAvgCDCAr[slice][2].filterPointsMedian(mCutDCA, mCutRMS);
          mBufferDCA.mDCAr_comb_C_Median[slice] = std::get<0>(dcaCrComb);
          mBufferDCA.mDCAr_comb_C_RMS[slice] = std::get<2>(dcaCrComb);

          const auto dcaCzComb = mAvgCDCAz[slice][2].filterPointsMedian(mCutDCA, mCutRMS);
          mBufferDCA.mDCAz_comb_C_Median[slice] = std::get<0>(dcaCzComb);
          mBufferDCA.mDCAz_comb_C_RMS[slice] = std::get<2>(dcaCzComb);
        }
      }
    }

    // calculate matching eff
    for (const auto& vals : mBufferVals) {
      const auto& val = vals.front();
      const auto nPoints = val.side.size();
      for (int i = 0; i < nPoints; ++i) {
        const auto tglBin = val.tglBin[i];
        const auto phiBin = val.phiBin[i];
        const auto qPtBin = val.qPtBin[i];
        const auto multBin = val.multBin[i];
        const auto dcar = val.dcar[i];
        const auto dcaz = val.dcaz[i];
        const auto hasITS = val.hasITS[i];
        const auto chi2Match = val.chi2Match[i];
        const auto dedxRatioqMax = val.dedxRatioqMax[i];
        const auto dedxRatioqTot = val.dedxRatioqTot[i];
        const auto sqrtChi2TPC = val.sqrtChi2TPC[i];
        const auto nClTPC = val.nClTPC[i];
        const int binInt = nBins - 1;
        const Side side = val.side[i];
        const bool isCSide = (side == Side::C);
        const auto& bufferDCARMSR = isCSide ? mBufferDCA.mTSTPC.mDCAr_C_RMS : mBufferDCA.mTSTPC.mDCAr_A_RMS;
        const auto& bufferDCARMSZ = isCSide ? mBufferDCA.mTSTPC.mDCAz_C_RMS : mBufferDCA.mTSTPC.mDCAz_A_RMS;
        const auto& bufferDCAMedR = isCSide ? mBufferDCA.mTSTPC.mDCAr_C_Median : mBufferDCA.mTSTPC.mDCAr_A_Median;
        const auto& bufferDCAMedZ = isCSide ? mBufferDCA.mTSTPC.mDCAz_C_Median : mBufferDCA.mTSTPC.mDCAz_A_Median;
        auto& mAvgEff = isCSide ? mAvgMeffC : mAvgMeffA;
        auto& mAvgChi2Match = isCSide ? mAvgChi2MatchC : mAvgChi2MatchA;
        auto& mAvgmMIPdEdxRatioqMax = isCSide ? mMIPdEdxRatioQMaxC : mMIPdEdxRatioQMaxA;
        auto& mAvgmMIPdEdxRatioqTot = isCSide ? mMIPdEdxRatioQTotC : mMIPdEdxRatioQTotA;
        auto& mAvgmTPCChi2 = isCSide ? mTPCChi2C : mTPCChi2A;
        auto& mAvgmTPCNCl = isCSide ? mTPCNClC : mTPCNClA;
        auto& mAvgmdEdxRatioQMax = isCSide ? mLogdEdxQMaxC : mLogdEdxQMaxA;
        auto& mAvgmdEdxRatioQTot = isCSide ? mLogdEdxQTotC : mLogdEdxQTotA;
        auto& mITSProperties = isCSide ? mITSPropertiesC : mITSPropertiesA;
        auto& mSigmaYZ = isCSide ? mSigmaYZC : mSigmaYZA;
        auto& mITSTPCDeltaP = isCSide ? mITSTPCDeltaPC : mITSTPCDeltaPA;

        const std::array<int, 5> bins{tglBin, phiBin, qPtBin, multBin, binInt};
        // fill bins
        for (auto bin : bins) {
          // make DCA cut - select only good tracks
          if ((std::abs(dcar - bufferDCAMedR[bin]) < (bufferDCARMSR[bin] * mCutRMS)) && (std::abs(dcaz - bufferDCAMedZ[bin]) < (bufferDCARMSZ[bin] * mCutRMS))) {
            const auto gID = val.gID[i];
            mAvgEff[bin][0].addValue(hasITS);
            // count tpc only tracks not matched
            if (!hasITS) {
              mAvgEff[bin][1].addValue(hasITS);
              mAvgEff[bin][2].addValue(hasITS);
            }
            // count tracks from ITS standalone and afterburner
            if (gID == o2::dataformats::GlobalTrackID::Source::ITS) {
              mAvgEff[bin][1].addValue(hasITS);
            } else if (gID == o2::dataformats::GlobalTrackID::Source::ITSAB) {
              mAvgEff[bin][2].addValue(hasITS);
            }
            if (chi2Match > 0) {
              mAvgChi2Match[bin][0].addValue(chi2Match);
              if (gID == o2::dataformats::GlobalTrackID::Source::ITS) {
                mAvgChi2Match[bin][1].addValue(chi2Match);
              } else if (gID == o2::dataformats::GlobalTrackID::Source::ITSAB) {
                mAvgChi2Match[bin][2].addValue(chi2Match);
              }
            }
            if (dedxRatioqMax > 0) {
              mAvgmMIPdEdxRatioqMax[bin][0].addValue(dedxRatioqMax);
            }
            if (dedxRatioqTot > 0) {
              mAvgmMIPdEdxRatioqTot[bin][0].addValue(dedxRatioqTot);
            }
            mAvgmTPCChi2[bin][0].addValue(sqrtChi2TPC);
            mAvgmTPCNCl[bin][0].addValue(nClTPC);
            if (hasITS) {
              if (dedxRatioqMax > 0) {
                mAvgmMIPdEdxRatioqMax[bin][1].addValue(dedxRatioqMax);
              }
              if (dedxRatioqTot > 0) {
                mAvgmMIPdEdxRatioqTot[bin][1].addValue(dedxRatioqTot);
              }
              mAvgmTPCChi2[bin][1].addValue(sqrtChi2TPC);
              mAvgmTPCNCl[bin][1].addValue(nClTPC);
            }

            float dedxNormQMax = val.dedxValsqMax[i].dedxNorm;
            if (dedxNormQMax > 0) {
              mAvgmdEdxRatioQMax[bin][0].addValue(dedxNormQMax);
            }

            float dedxNormQTot = val.dedxValsqTot[i].dedxNorm;
            if (dedxNormQTot > 0) {
              mAvgmdEdxRatioQTot[bin][0].addValue(dedxNormQTot);
            }

            float dedxIROCQMax = val.dedxValsqMax[i].dedxIROC;
            if (dedxIROCQMax > 0) {
              mAvgmdEdxRatioQMax[bin][1].addValue(dedxIROCQMax);
            }

            float dedxIROCQTot = val.dedxValsqTot[i].dedxIROC;
            if (dedxIROCQTot > 0) {
              mAvgmdEdxRatioQTot[bin][1].addValue(dedxIROCQTot);
            }

            float dedxOROC1QMax = val.dedxValsqMax[i].dedxOROC1;
            if (dedxOROC1QMax > 0) {
              mAvgmdEdxRatioQMax[bin][2].addValue(dedxOROC1QMax);
            }

            float dedxOROC1QTot = val.dedxValsqTot[i].dedxOROC1;
            if (dedxOROC1QTot > 0) {
              mAvgmdEdxRatioQTot[bin][2].addValue(dedxOROC1QTot);
            }

            float dedxOROC2QMax = val.dedxValsqMax[i].dedxOROC2;
            if (dedxOROC2QMax > 0) {
              mAvgmdEdxRatioQMax[bin][3].addValue(dedxOROC2QMax);
            }

            float dedxOROC2QTot = val.dedxValsqTot[i].dedxOROC2;
            if (dedxOROC2QTot > 0) {
              mAvgmdEdxRatioQTot[bin][3].addValue(dedxOROC2QTot);
            }

            float dedxOROC3QMax = val.dedxValsqMax[i].dedxOROC3;
            if (dedxOROC3QMax > 0) {
              mAvgmdEdxRatioQMax[bin][4].addValue(dedxOROC3QMax);
            }

            float dedxOROC3QTot = val.dedxValsqTot[i].dedxOROC3;
            if (dedxOROC3QTot > 0) {
              mAvgmdEdxRatioQTot[bin][4].addValue(dedxOROC3QTot);
            }

            float nClITS = val.nClITS[i];
            if (nClITS > 0) {
              mITSProperties[bin][0].addValue(nClITS);
            }
            float chi2ITS = val.chi2ITS[i];
            if (chi2ITS > 0) {
              mITSProperties[bin][1].addValue(chi2ITS);
            }

            float sigmay2 = val.sigmaY2[i];
            if (sigmay2 > 0) {
              mSigmaYZ[bin][0].addValue(sigmay2);
            }
            float sigmaz2 = val.sigmaZ2[i];
            if (sigmaz2 > 0) {
              mSigmaYZ[bin][1].addValue(sigmaz2);
            }

            float deltaP2 = val.deltaP2[i];
            if (deltaP2 != -999) {
              mITSTPCDeltaP[bin][0].addValue(deltaP2);
            }

            float deltaP3 = val.deltaP3[i];
            if (deltaP3 != -999) {
              mITSTPCDeltaP[bin][1].addValue(deltaP3);
            }

            float deltaP4 = val.deltaP4[i];
            if (deltaP4 != -999) {
              mITSTPCDeltaP[bin][2].addValue(deltaP4);
            }
          }
        }
      }
    }

    // store matching eff
    for (int slice = 0; slice < nBins; ++slice) {
      for (int i = 0; i < mAvgMeffA[slice].size(); ++i) {
        auto& itsBuf = (i == 0) ? mBufferDCA.mITSTPCAll : ((i == 1) ? mBufferDCA.mITSTPCStandalone : mBufferDCA.mITSTPCAfterburner);
        itsBuf.mITSTPC_A_MatchEff[slice] = mAvgMeffA[slice][i].getMean();
        itsBuf.mITSTPC_C_MatchEff[slice] = mAvgMeffC[slice][i].getMean();
        itsBuf.mITSTPC_A_Chi2Match[slice] = mAvgChi2MatchA[slice][i].getMean();
        itsBuf.mITSTPC_C_Chi2Match[slice] = mAvgChi2MatchC[slice][i].getMean();
      }

      // loop over TPC and ITS-TPC tracks
      for (int i = 0; i < mMIPdEdxRatioQMaxC[slice].size(); ++i) {
        auto& buff = (i == 0) ? mBufferDCA.mTSTPC : mBufferDCA.mTSITSTPC;
        buff.mMIPdEdxRatioQMaxA[slice] = mMIPdEdxRatioQMaxA[slice][i].getMean();
        buff.mMIPdEdxRatioQMaxC[slice] = mMIPdEdxRatioQMaxC[slice][i].getMean();
        buff.mMIPdEdxRatioQTotA[slice] = mMIPdEdxRatioQTotA[slice][i].getMean();
        buff.mMIPdEdxRatioQTotC[slice] = mMIPdEdxRatioQTotC[slice][i].getMean();
        buff.mTPCChi2C[slice] = mTPCChi2C[slice][i].getMean();
        buff.mTPCChi2A[slice] = mTPCChi2A[slice][i].getMean();
        buff.mTPCNClC[slice] = mTPCNClC[slice][i].getMean();
        buff.mTPCNClA[slice] = mTPCNClA[slice][i].getMean();
      }

      // loop over qMax and qTot
      for (int type = 0; type < 2; ++type) {
        auto& logdEdxA = (type == 0) ? mLogdEdxQMaxA : mLogdEdxQTotA;
        auto& buffer = (type == 0) ? mBufferDCA.mdEdxQMax : mBufferDCA.mdEdxQTot;
        // fill A-side
        buffer.mLogdEdx_A_Median[slice] = logdEdxA[slice][0].getMedian();
        buffer.mLogdEdx_A_RMS[slice] = logdEdxA[slice][0].getStdDev();
        buffer.mLogdEdx_A_IROC_Median[slice] = logdEdxA[slice][1].getMedian();
        buffer.mLogdEdx_A_IROC_RMS[slice] = logdEdxA[slice][1].getStdDev();
        buffer.mLogdEdx_A_OROC1_Median[slice] = logdEdxA[slice][2].getMedian();
        buffer.mLogdEdx_A_OROC1_RMS[slice] = logdEdxA[slice][2].getStdDev();
        buffer.mLogdEdx_A_OROC2_Median[slice] = logdEdxA[slice][3].getMedian();
        buffer.mLogdEdx_A_OROC2_RMS[slice] = logdEdxA[slice][3].getStdDev();
        buffer.mLogdEdx_A_OROC3_Median[slice] = logdEdxA[slice][4].getMedian();
        buffer.mLogdEdx_A_OROC3_RMS[slice] = logdEdxA[slice][4].getStdDev();
        // fill C-side
        auto& logdEdxC = (type == 0) ? mLogdEdxQMaxC : mLogdEdxQTotC;
        buffer.mLogdEdx_C_Median[slice] = logdEdxC[slice][0].getMedian();
        buffer.mLogdEdx_C_RMS[slice] = logdEdxC[slice][0].getStdDev();
        buffer.mLogdEdx_C_IROC_Median[slice] = logdEdxC[slice][1].getMedian();
        buffer.mLogdEdx_C_IROC_RMS[slice] = logdEdxC[slice][1].getStdDev();
        buffer.mLogdEdx_C_OROC1_Median[slice] = logdEdxC[slice][2].getMedian();
        buffer.mLogdEdx_C_OROC1_RMS[slice] = logdEdxC[slice][2].getStdDev();
        buffer.mLogdEdx_C_OROC2_Median[slice] = logdEdxC[slice][3].getMedian();
        buffer.mLogdEdx_C_OROC2_RMS[slice] = logdEdxC[slice][3].getStdDev();
        buffer.mLogdEdx_C_OROC3_Median[slice] = logdEdxC[slice][4].getMedian();
        buffer.mLogdEdx_C_OROC3_RMS[slice] = logdEdxC[slice][4].getStdDev();
      }

      // ITS properties
      // A-side
      mBufferDCA.mITS_A_NCl_Median[slice] = mITSPropertiesA[slice][0].getMedian();
      mBufferDCA.mITS_A_NCl_RMS[slice] = mITSPropertiesA[slice][0].getStdDev();
      mBufferDCA.mSqrtITSChi2_Ncl_A_Median[slice] = mITSPropertiesA[slice][1].getMedian();
      mBufferDCA.mSqrtITSChi2_Ncl_A_RMS[slice] = mITSPropertiesA[slice][1].getStdDev();
      // C-side
      mBufferDCA.mITS_C_NCl_Median[slice] = mITSPropertiesC[slice][0].getMedian();
      mBufferDCA.mITS_C_NCl_RMS[slice] = mITSPropertiesC[slice][0].getStdDev();
      mBufferDCA.mSqrtITSChi2_Ncl_C_Median[slice] = mITSPropertiesC[slice][1].getMedian();
      mBufferDCA.mSqrtITSChi2_Ncl_C_RMS[slice] = mITSPropertiesC[slice][1].getStdDev();

      //...
      mBufferDCA.mITSTPCDeltaP2_A_Median[slice] = mITSTPCDeltaPA[slice][0].getMedian();
      mBufferDCA.mITSTPCDeltaP3_A_Median[slice] = mITSTPCDeltaPA[slice][1].getMedian();
      mBufferDCA.mITSTPCDeltaP4_A_Median[slice] = mITSTPCDeltaPA[slice][2].getMedian();
      mBufferDCA.mITSTPCDeltaP2_C_Median[slice] = mITSTPCDeltaPC[slice][0].getMedian();
      mBufferDCA.mITSTPCDeltaP3_C_Median[slice] = mITSTPCDeltaPC[slice][1].getMedian();
      mBufferDCA.mITSTPCDeltaP4_C_Median[slice] = mITSTPCDeltaPC[slice][2].getMedian();
      mBufferDCA.mITSTPCDeltaP2_A_RMS[slice] = mITSTPCDeltaPA[slice][0].getStdDev();
      mBufferDCA.mITSTPCDeltaP3_A_RMS[slice] = mITSTPCDeltaPA[slice][1].getStdDev();
      mBufferDCA.mITSTPCDeltaP4_A_RMS[slice] = mITSTPCDeltaPA[slice][2].getStdDev();
      mBufferDCA.mITSTPCDeltaP2_C_RMS[slice] = mITSTPCDeltaPC[slice][0].getStdDev();
      mBufferDCA.mITSTPCDeltaP3_C_RMS[slice] = mITSTPCDeltaPC[slice][1].getStdDev();
      mBufferDCA.mITSTPCDeltaP4_C_RMS[slice] = mITSTPCDeltaPC[slice][2].getStdDev();
      mBufferDCA.mTPCSigmaY2A_Median[slice] = mSigmaYZA[slice][0].getMedian();
      mBufferDCA.mTPCSigmaZ2A_Median[slice] = mSigmaYZA[slice][1].getMedian();
      mBufferDCA.mTPCSigmaY2C_Median[slice] = mSigmaYZC[slice][0].getMedian();
      mBufferDCA.mTPCSigmaZ2C_Median[slice] = mSigmaYZC[slice][1].getMedian();
      mBufferDCA.mTPCSigmaY2A_RMS[slice] = mSigmaYZA[slice][0].getStdDev();
      mBufferDCA.mTPCSigmaZ2A_RMS[slice] = mSigmaYZA[slice][1].getStdDev();
      mBufferDCA.mTPCSigmaY2C_RMS[slice] = mSigmaYZC[slice][0].getStdDev();
      mBufferDCA.mTPCSigmaZ2C_RMS[slice] = mSigmaYZC[slice][1].getStdDev();
    }

    auto stop = timer::now();
    std::chrono::duration<float> time = stop - startTotal;
    LOGP(info, "Time series creation took {}", time.count());

    // send data
    sendOutput(pc);
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    for (auto& streamer : mStreamer) {
      streamer->Close();
    }
    eos.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    mTPCVDriftHelper.accountCCDBInputs(matcher, obj);
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 private:
  /// buffer struct for multithreading
  struct ValsdEdx {
    float dedxNorm = 0;
    float dedxIROC = 0;
    float dedxOROC1 = 0;
    float dedxOROC2 = 0;
    float dedxOROC3 = 0;
  };

  struct FillVals {
    void reserve(int n, int type)
    {
      side.reserve(n);
      tglBin.reserve(n);
      phiBin.reserve(n);
      qPtBin.reserve(n);
      multBin.reserve(n);
      dcar.reserve(n);
      dcaz.reserve(n);
      dcarW.reserve(n);
      dedxRatioqTot.reserve(n);
      dedxRatioqMax.reserve(n);
      sqrtChi2TPC.reserve(n);
      nClTPC.reserve(n);
      if (type == 1) {
        hasITS.reserve(n);
        chi2Match.reserve(n);
        gID.reserve(n);
        dedxValsqTot.reserve(n);
        dedxValsqMax.reserve(n);
        nClITS.reserve(n);
        chi2ITS.reserve(n);
        deltaP2.reserve(n);
        deltaP3.reserve(n);
        deltaP4.reserve(n);
        sigmaY2.reserve(n);
        sigmaZ2.reserve(n);
      } else if (type == 0) {
        dcarcomb.reserve(n);
        dcazcomb.reserve(n);
      }
    }

    void clear()
    {
      side.clear();
      tglBin.clear();
      phiBin.clear();
      qPtBin.clear();
      multBin.clear();
      dcar.clear();
      dcaz.clear();
      dcarW.clear();
      hasITS.clear();
      chi2Match.clear();
      dedxRatioqTot.clear();
      dedxRatioqMax.clear();
      sqrtChi2TPC.clear();
      nClTPC.clear();
      gID.clear();
      dcarcomb.clear();
      dcazcomb.clear();
      nClITS.clear();
      chi2ITS.clear();
      dedxValsqTot.clear();
      dedxValsqMax.clear();
      deltaP2.clear();
      deltaP3.clear();
      deltaP4.clear();
      sigmaY2.clear();
      sigmaZ2.clear();
    }

    void emplace_back(Side sideTmp, int tglBinTmp, int phiBinTmp, int qPtBinTmp, int multBinTmp, float dcarTmp, float dcazTmp, float dcarWTmp, float dedxRatioqTotTmp, float dedxRatioqMaxTmp, float sqrtChi2TPCTmp, float nClTPCTmp, o2::dataformats::GlobalTrackID::Source gIDTmp, float chi2MatchTmp, int hasITSTmp, int nClITSTmp, float chi2ITSTmp, const ValsdEdx& dedxValsqTotTmp, const ValsdEdx& dedxValsqMaxTmp, float sigmaY2Tmp, float sigmaZ2Tmp)
    {
      side.emplace_back(sideTmp);
      tglBin.emplace_back(tglBinTmp);
      phiBin.emplace_back(phiBinTmp);
      qPtBin.emplace_back(qPtBinTmp);
      multBin.emplace_back(multBinTmp);
      dcar.emplace_back(dcarTmp);
      dcaz.emplace_back(dcazTmp);
      dcarW.emplace_back(dcarWTmp);
      dedxRatioqTot.emplace_back(dedxRatioqTotTmp);
      dedxRatioqMax.emplace_back(dedxRatioqMaxTmp);
      sqrtChi2TPC.emplace_back(sqrtChi2TPCTmp);
      nClTPC.emplace_back(nClTPCTmp);
      chi2Match.emplace_back(chi2MatchTmp);
      hasITS.emplace_back(hasITSTmp);
      gID.emplace_back(gIDTmp);
      dedxValsqMax.emplace_back(dedxValsqMaxTmp);
      dedxValsqTot.emplace_back(dedxValsqTotTmp);
      nClITS.emplace_back(nClITSTmp);
      chi2ITS.emplace_back(chi2ITSTmp);
      sigmaY2.emplace_back(sigmaY2Tmp);
      sigmaZ2.emplace_back(sigmaZ2Tmp);
      deltaP2.emplace_back(-999);
      deltaP3.emplace_back(-999);
      deltaP4.emplace_back(-999);
    }

    void setDeltaParam(float deltaP2Tmp, float deltaP3Tmp, float deltaP4Tmp)
    {
      if (!deltaP2.empty()) {
        deltaP2.back() = deltaP2Tmp;
        deltaP3.back() = deltaP3Tmp;
        deltaP4.back() = deltaP4Tmp;
      }
    }

    void emplace_back_ITSTPC(Side sideTmp, int tglBinTmp, int phiBinTmp, int qPtBinTmp, int multBinTmp, float dcarTmp, float dcazTmp, float dcarWTmp, float dedxRatioqTotTmp, float dedxRatioqMaxTmp, float sqrtChi2TPCTmp, float nClTPCTmp, float dcarCombTmp, float dcazCombTmp)
    {
      side.emplace_back(sideTmp);
      tglBin.emplace_back(tglBinTmp);
      phiBin.emplace_back(phiBinTmp);
      qPtBin.emplace_back(qPtBinTmp);
      multBin.emplace_back(multBinTmp);
      dcar.emplace_back(dcarTmp);
      dcaz.emplace_back(dcazTmp);
      dcarW.emplace_back(dcarWTmp);
      dedxRatioqTot.emplace_back(dedxRatioqTotTmp);
      dedxRatioqMax.emplace_back(dedxRatioqMaxTmp);
      sqrtChi2TPC.emplace_back(sqrtChi2TPCTmp);
      nClTPC.emplace_back(nClTPCTmp);
      dcarcomb.emplace_back(dcarCombTmp);
      dcazcomb.emplace_back(dcazCombTmp);
    }

    std::vector<Side> side;
    std::vector<int> tglBin;
    std::vector<int> phiBin;
    std::vector<int> qPtBin;
    std::vector<int> multBin;
    std::vector<float> dcar;
    std::vector<float> dcaz;
    std::vector<float> dcarW;
    std::vector<bool> hasITS;
    std::vector<float> chi2Match;
    std::vector<float> dedxRatioqTot;
    std::vector<float> dedxRatioqMax;
    std::vector<float> sqrtChi2TPC;
    std::vector<float> nClTPC;
    std::vector<float> dcarcomb;
    std::vector<float> dcazcomb;
    std::vector<ValsdEdx> dedxValsqTot;
    std::vector<ValsdEdx> dedxValsqMax;
    std::vector<int> nClITS;
    std::vector<float> chi2ITS;
    std::vector<o2::dataformats::GlobalTrackID::Source> gID;
    std::vector<float> deltaP2;
    std::vector<float> deltaP3;
    std::vector<float> deltaP4;
    std::vector<float> sigmaY2;
    std::vector<float> sigmaZ2;
  };
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                  ///< info for CCDB request
  const bool mDisableWriter{false};                                        ///< flag if no ROOT output will be written
  o2::base::Propagator::MatCorrType mMatType;                              ///< material for propagation
  const bool mUnbinnedWriter{false};                                       /// write out additional unbinned data
  const bool mTPCOnly{false};                                              ///< produce only TPC variables
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;           ///< steers the input
  int mPhiBins = SECTORSPERSIDE;                                           ///< number of phi bins
  int mTglBins{3};                                                         ///< number of tgl bins
  int mQPtBins{20};                                                        ///< number of qPt bins
  TimeSeriesITSTPC mBufferDCA;                                             ///< buffer for integrate DCAs
  std::vector<std::array<RobustAverage, 3>> mAvgADCAr;                     ///< for averaging the DCAr for TPC and ITS-TPC tracks for A-side
  std::vector<std::array<RobustAverage, 3>> mAvgCDCAr;                     ///< for averaging the DCAr for TPC and ITS-TPC tracks for C-side
  std::vector<std::array<RobustAverage, 3>> mAvgADCAz;                     ///< for averaging the DCAz for TPC and ITS-TPC tracks for A-side
  std::vector<std::array<RobustAverage, 3>> mAvgCDCAz;                     ///< for averaging the DCAz for TPC and ITS-TPC tracks for C-side
  std::vector<std::array<RobustAverage, 2>> mMIPdEdxRatioQMaxA;            ///< for averaging MIP/dEdx - qMax -
  std::vector<std::array<RobustAverage, 2>> mMIPdEdxRatioQMaxC;            ///< for averaging MIP/dEdx - qMax -
  std::vector<std::array<RobustAverage, 2>> mMIPdEdxRatioQTotA;            ///< for averaging MIP/dEdx - qTot -
  std::vector<std::array<RobustAverage, 2>> mMIPdEdxRatioQTotC;            ///< for averaging MIP/dEdx - qTot -
  std::vector<std::array<RobustAverage, 2>> mTPCChi2A;                     ///< for averaging chi2 TPC A
  std::vector<std::array<RobustAverage, 2>> mTPCChi2C;                     ///< for averaging chi2 TPC C
  std::vector<std::array<RobustAverage, 2>> mTPCNClA;                      ///< for averaging number of cluster A
  std::vector<std::array<RobustAverage, 2>> mTPCNClC;                      ///< for averaging number of cluster C
  std::vector<std::array<RobustAverage, 3>> mAvgMeffA;                     ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 3>> mAvgMeffC;                     ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 3>> mAvgChi2MatchA;                ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 3>> mAvgChi2MatchC;                ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 10>> mLogdEdxQTotA;                ///< for log dedx A side - qTot
  std::vector<std::array<RobustAverage, 10>> mLogdEdxQTotC;                ///< for log dedx C side - qTot
  std::vector<std::array<RobustAverage, 10>> mLogdEdxQMaxA;                ///< for log dedx A side - qMax
  std::vector<std::array<RobustAverage, 10>> mLogdEdxQMaxC;                ///< for log dedx C side - qMax
  std::vector<std::array<RobustAverage, 2>> mITSPropertiesA;               ///< mITS_NCl, mSqrtITSChi2_Ncl, mSqrtMatchChi2
  std::vector<std::array<RobustAverage, 2>> mITSPropertiesC;               ///< mITS_NCl, mSqrtITSChi2_Ncl, mSqrtMatchChi2
  std::vector<std::array<RobustAverage, 3>> mITSTPCDeltaPA;                ///< delta parameters 2,3,4
  std::vector<std::array<RobustAverage, 3>> mITSTPCDeltaPC;                ///< delta parameters 2,3,4
  std::vector<std::array<RobustAverage, 2>> mSigmaYZA;                     ///< sigmaY2 and sigmaZ2 A-side
  std::vector<std::array<RobustAverage, 2>> mSigmaYZC;                     ///< sigmaY2 and sigmaZ2 C-side
  int mNMaxTracks{-1};                                                     ///< maximum number of tracks to process
  float mMinMom{1};                                                        ///< minimum accepted momentum
  int mMinNCl{80};                                                         ///< minimum accepted number of clusters per track
  float mMaxTgl{1};                                                        ///< maximum eta
  float mMaxQPt{5};                                                        ///< max qPt bin
  float mCoarseStep{1};                                                    ///< coarse step during track propagation
  float mFineStep{0.005};                                                  ///< fine step during track propagation
  float mCutDCA{5};                                                        ///< cut on the abs(DCA-median)
  float mCutRMS{5};                                                        ///< sigma cut for mean,median calculation
  float mRefXSec{108.475};                                                 ///< reference lx position for sector information (default centre of IROC)
  int mNThreads{1};                                                        ///< number of parallel threads
  float maxITSTPCDCAr{0.2};                                                ///< maximum abs DCAr value for ITS-TPC tracks
  float maxITSTPCDCAz{10};                                                 ///< maximum abs DCAz value for ITS-TPC tracks
  float maxITSTPCDCAr_comb{0.2};                                           ///< max abs DCA for ITS-TPC DCA to vertex
  float maxITSTPCDCAz_comb{0.2};                                           ///< max abs DCA for ITS-TPC DCA to vertex
  gsl::span<const TPCClRefElem> mTPCTrackClIdx{};                          ///< cluster refs for debugging
  std::vector<std::array<FillVals, 2>> mBufferVals;                        ///< buffer for multithreading
  uint32_t mFirstTFOrbit{0};                                               ///< first TF orbit
  float mTimeWindowMUS{50};                                                ///< time window in mus for local mult estimate
  float mMIPdEdx{50};                                                      ///< MIP dEdx position for MIP/dEdx monitoring
  std::vector<int> mNTracksWindow;                                         ///< number of tracks in time window
  std::vector<int> mNearestVtxTPC;                                         ///< nearest vertex for tpc tracks
  o2::tpc::VDriftHelper mTPCVDriftHelper{};                                ///< helper for v-drift
  float mVDrift{2.64};                                                     ///< v drift in mus
  float mMaxSnp{0.85};                                                     ///< max sinus phi for propagation
  float mXCoarse{40};                                                      ///< perform propagation with coarse steps up to this mx
  float mSqrt{13600};                                                      ///< centre of mass energy
  int mMultBins{20};                                                       ///< multiplicity bins
  int mMultMax{80000};                                                     ///< maximum multiplicity
  PIDResponse mPID;                                                        ///< PID response
  int mMinTracksPerVertex{5};                                              ///< minimum number of tracks per vertex
  float mMaxdEdxRatio{0.3};                                                ///< maximum abs dedx ratio: log(dedx_exp(pion)/dedx)
  float mMaxdEdxRegionRatio{0.5};                                          ///< maximum abs dedx region ratio: log(dedx_region/dedx)
  float mSamplingFactor{0.1};                                              ///< sampling factor in case sampling is used for unbinned data
  bool mSampleTsallis{false};                                              ///< perform sampling of unbinned data
  std::mt19937 mGenerator;                                                 ///< random generator for debug tree sampling
  std::vector<std::unique_ptr<o2::utils::TreeStreamRedirector>> mStreamer; ///< streamer for unbinned data

  /// check if track passes coarse cuts
  bool acceptTrack(const TrackTPC& track) const
  {
    if ((track.getP() < mMinMom) || (track.getNClusters() < mMinNCl) || std::abs(track.getTgl()) > mMaxTgl) {
      return false;
    }
    return true;
  }

  void fillDCA(const gsl::span<const TrackTPC> tracksTPC, const gsl::span<const o2::dataformats::TrackTPCITS> tracksITSTPC, const gsl::span<const o2::dataformats::PrimaryVertex> vertices, const int iTrk, const int iThread, const std::unordered_map<unsigned int, std::array<int, 2>>& indicesITSTPC, const gsl::span<const o2::its::TrackITS> tracksITS, const std::vector<std::tuple<int, float, float>>& idxTPCTrackToTOFCluster, const gsl::span<const o2::tof::Cluster> tofClusters)
  {
    o2::track::TrackParCov track = tracksTPC[iTrk];
    const auto& trackFull = tracksTPC[iTrk];

    // propagate track to the DCA and fill in slice
    auto propagator = o2::base::Propagator::Instance();

    // propagate track to DCA
    o2::gpu::gpustd::array<float, 2> dca;
    const o2::math_utils::Point3D<float> refPoint{0, 0, 0};

    // coarse propagation
    if (!propagator->PropagateToXBxByBz(track, mXCoarse, mMaxSnp, mCoarseStep, mMatType)) {
      return;
    }

    // fine propagation with Bz only
    if (!propagator->propagateToDCA(refPoint, track, propagator->getNominalBz(), mFineStep, mMatType, &dca)) {
      return;
    }

    o2::track::TrackPar trackTmp(tracksTPC[iTrk]);

    // coarse propagation to centre of IROC for phi bin
    if (!propagator->propagateTo(trackTmp, mRefXSec, false, mMaxSnp, mCoarseStep, mMatType)) {
      return;
    }

    const int tglBin = mTglBins * std::abs(trackTmp.getTgl()) / mMaxTgl + mPhiBins;
    const int phiBin = mPhiBins * trackTmp.getPhi() / o2::constants::math::TwoPI;

    const int offsQPtBin = mPhiBins + mTglBins;
    const int qPtBin = offsQPtBin + mQPtBins * (trackTmp.getQ2Pt() + mMaxQPt) / (2 * mMaxQPt);
    const int localMult = mNTracksWindow[iTrk];

    const int offsMult = offsQPtBin + mQPtBins;
    const int multBin = offsMult + mMultBins * localMult / mMultMax;
    const int nBins = getNBins();

    if ((phiBin < 0) || (phiBin > mPhiBins) || (tglBin < mPhiBins) || (tglBin > offsQPtBin) || (qPtBin < offsQPtBin) || (qPtBin > offsMult) || (multBin < offsMult) || (multBin > offsMult + mMultBins)) {
      return;
    }

    float sigmaY2 = 0;
    float sigmaZ2 = 0;
    const int sector = o2::math_utils::angle2Sector(trackTmp.getPhiPos());
    // find possible ITS-TPC track and vertex index
    auto it = indicesITSTPC.find(iTrk);
    const auto idxITSTPC = (it != indicesITSTPC.end()) ? (it->second) : std::array<int, 2>{-1, -1};

    // get vertex (check if vertex ID is valid). In case no vertex is assigned return nearest vertex or else default vertex
    const auto vertex = (idxITSTPC.back() != -1) ? vertices[idxITSTPC.back()] : ((mNearestVtxTPC[iTrk] != -1) ? vertices[mNearestVtxTPC[iTrk]] : o2::dataformats::PrimaryVertex{});

    // calculate DCAz: (time TPC track - time vertex) * vDrift + sign_side * vertexZ
    const float signSide = trackFull.hasCSideClustersOnly() ? -1 : 1; // invert sign for C-side
    const float dcaZFromDeltaTime = (vertex.getTimeStamp().getTimeStamp() == 0) ? 0 : (o2::tpc::ParameterElectronics::Instance().ZbinWidth * trackFull.getTime0() - vertex.getTimeStamp().getTimeStamp()) * mVDrift + signSide * vertex.getZ();

    // for weight of DCA
    const float resCl = std::min(trackFull.getNClusters(), static_cast<int>(Mapper::PADROWS)) / static_cast<float>(Mapper::PADROWS);

    const float div = (resCl * track.getPt());
    if (div == 0) {
      return;
    }

    const float fB = 0.2 / div;
    const float fA = 0.15 + 0.15;                          //  = 0.15 with additional misalignment error
    const float dcarW = 1. / std::sqrt(fA * fA + fB * fB); // Weight of DCA: Rms2 ~ 0.15^2 +  k/(L^2*pt) →    0.15**2 +  (0.2/((NCl/152)*pt)^2);

    // store values for TPC DCA only for A- or C-side only tracks
    const bool hasITSTPC = idxITSTPC.front() != -1;

    // get ratio of chi2 in case ITS-TPC track has been found
    const float chi2 = hasITSTPC ? tracksITSTPC[idxITSTPC.front()].getChi2Match() : -1;
    auto gID = o2::dataformats::GlobalTrackID::Source::TPC; // source
    // check for source in case of ITS-TPC
    if (hasITSTPC) {
      const auto src = tracksITSTPC[idxITSTPC.front()].getRefITS().getSource();
      if (src == o2::dataformats::GlobalTrackID::ITS) {
        gID = o2::dataformats::GlobalTrackID::Source::ITS;
      } else if (src == o2::dataformats::GlobalTrackID::ITSAB) {
        gID = o2::dataformats::GlobalTrackID::Source::ITSAB;
      }
    }

    const float chi2Match = (chi2 > 0) ? std::sqrt(chi2) : -1;
    const float sqrtChi2TPC = (trackFull.getChi2() > 0) ? std::sqrt(trackFull.getChi2()) : 0;
    const float nClTPC = trackFull.getNClusters();

    // const float dedx = mUseQMax ? track.getdEdx().dEdxMaxTPC : track.getdEdx().dEdxTotTPC;
    const float dedxRatioqTot = (trackFull.getdEdx().dEdxTotTPC > 0) ? (mMIPdEdx / trackFull.getdEdx().dEdxTotTPC) : -1;
    const float dedxRatioqMax = (trackFull.getdEdx().dEdxMaxTPC > 0) ? (mMIPdEdx / trackFull.getdEdx().dEdxMaxTPC) : -1;

    const auto dedxQTotVars = getdEdxVars(0, trackFull);
    const auto dedxQMaxVars = getdEdxVars(1, trackFull);

    // make check to avoid crash in case no or less ITS tracks have been found!
    const int idxITSTrack = (hasITSTPC && (gID == o2::dataformats::GlobalTrackID::Source::ITS)) ? tracksITSTPC[idxITSTPC.front()].getRefITS().getIndex() : -1;
    const bool idxITSCheck = (idxITSTrack != -1);

    const int nClITS = idxITSCheck ? tracksITS[idxITSTrack].getNClusters() : -1;
    float chi2ITS = idxITSCheck ? tracksITS[idxITSTrack].getChi2() : -1;
    if ((chi2ITS > 0) && (nClITS > 0)) {
      chi2ITS = std::sqrt(chi2ITS / nClITS);
    }
    sigmaY2 = track.getSigmaY2();
    sigmaZ2 = track.getSigmaZ2();
    if (trackFull.hasCSideClustersOnly()) {
      mBufferVals[iThread].front().emplace_back(Side::C, tglBin, phiBin, qPtBin, multBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatioqTot, dedxRatioqMax, sqrtChi2TPC, nClTPC, gID, chi2Match, hasITSTPC, nClITS, chi2ITS, dedxQTotVars, dedxQMaxVars, sigmaY2, sigmaZ2);
    } else if (trackFull.hasASideClustersOnly()) {
      mBufferVals[iThread].front().emplace_back(Side::A, tglBin, phiBin, qPtBin, multBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatioqTot, dedxRatioqMax, sqrtChi2TPC, nClTPC, gID, chi2Match, hasITSTPC, nClITS, chi2ITS, dedxQTotVars, dedxQMaxVars, sigmaY2, sigmaZ2);
    }

    // make propagation for ITS-TPC Track
    // check if the track was assigned to ITS track
    o2::gpu::gpustd::array<float, 2> dcaITSTPC{0, 0};
    float deltaP2 = -999;
    float deltaP3 = -999;
    float deltaP4 = -999;
    if (hasITSTPC) {
      // propagate ITS-TPC track to (0,0)
      auto trackITSTPCTmp = tracksITSTPC[idxITSTPC.front()];
      // fine propagation with Bz only
      if (propagator->propagateToDCA(refPoint, trackITSTPCTmp, propagator->getNominalBz(), mFineStep, mMatType, &dcaITSTPC)) {
        // make cut on abs(DCA)
        if ((std::abs(dcaITSTPC[0]) < maxITSTPCDCAr) && (std::abs(dcaITSTPC[1]) < maxITSTPCDCAz)) {
          // store TPC only DCAs
          // propagate to vertex in case the track belongs to vertex
          const bool contributeToVertex = (idxITSTPC.back() != -1);
          o2::gpu::gpustd::array<float, 2> dcaITSTPCTmp{-1, -1};

          if (contributeToVertex) {
            propagator->propagateToDCA(vertex.getXYZ(), trackITSTPCTmp, propagator->getNominalBz(), mFineStep, mMatType, &dcaITSTPCTmp);
          }

          // make cut around DCA to vertex due to gammas
          if ((std::abs(dcaITSTPCTmp[0]) < maxITSTPCDCAr_comb) && (std::abs(dcaITSTPCTmp[1]) < maxITSTPCDCAz_comb)) {
            // propagate TPC track to ITS-TPC track and store delta track parameters
            if (track.rotate(trackITSTPCTmp.getAlpha()) && propagator->propagateTo(track, trackITSTPCTmp.getX(), false, mMaxSnp, mFineStep, mMatType)) {
              deltaP2 = track.getParam(2) - trackITSTPCTmp.getParam(2);
              deltaP3 = track.getParam(3) - trackITSTPCTmp.getParam(3);
              deltaP4 = track.getParam(4) - trackITSTPCTmp.getParam(4);
              mBufferVals[iThread].front().setDeltaParam(deltaP2, deltaP3, deltaP4);
            }
          } else {
            dcaITSTPCTmp[0] = -1;
            dcaITSTPCTmp[1] = -1;
          }

          if (trackFull.hasCSideClustersOnly()) {
            mBufferVals[iThread].back().emplace_back_ITSTPC(Side::C, tglBin, phiBin, qPtBin, multBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatioqTot, dedxRatioqMax, sqrtChi2TPC, nClTPC, dcaITSTPCTmp[0], dcaITSTPCTmp[1]);
          } else if (trackFull.hasASideClustersOnly()) {
            mBufferVals[iThread].back().emplace_back_ITSTPC(Side::A, tglBin, phiBin, qPtBin, multBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatioqTot, dedxRatioqMax, sqrtChi2TPC, nClTPC, dcaITSTPCTmp[0], dcaITSTPCTmp[1]);
          }
        }
      }
    }

    if (mUnbinnedWriter && mStreamer[iThread]) {
      const float factorPt = mSamplingFactor;
      bool writeData = true;
      float weight = 0;
      if (mSampleTsallis) {
        std::uniform_real_distribution<> distr(0., 1.);
        writeData = o2::math_utils::Tsallis::downsampleTsallisCharged(tracksTPC[iTrk].getPt(), factorPt, mSqrt, weight, distr(mGenerator));
      }
      if (writeData) {
        auto clusterMask = makeClusterBitMask(trackFull);
        const auto& trkOrig = tracksTPC[iTrk];
        const bool isNearestVtx = (idxITSTPC.back() == -1); // is nearest vertex in case no vertex was found
        const float mx_ITS = hasITSTPC ? tracksITSTPC[idxITSTPC.front()].getX() : -1;
        const float pt_ITS = hasITSTPC ? tracksITSTPC[idxITSTPC.front()].getQ2Pt() : -1;
        const float chi2match_ITSTPC = hasITSTPC ? tracksITSTPC[idxITSTPC.front()].getChi2Match() : -1;
        const int nClITS = idxITSCheck ? tracksITS[idxITSTrack].getNClusters() : -1;
        const int chi2ITS = idxITSCheck ? tracksITS[idxITSTrack].getChi2() : -1;
        int typeSide = 2; // A- and C-Side cluster
        if (trackFull.hasASideClustersOnly()) {
          typeSide = 0;
        } else if (trackFull.hasCSideClustersOnly()) {
          typeSide = 1;
        }

        // check for TOF and propagate TPC track to TOF cluster
        bool hasTOFCluster = (std::get<0>(idxTPCTrackToTOFCluster[iTrk]) != -1);
        auto tofCl = hasTOFCluster ? tofClusters[std::get<0>(idxTPCTrackToTOFCluster[iTrk])] : o2::tof::Cluster();

        float tpcYDeltaAtTOF = -999;
        float tpcZDeltaAtTOF = -999;
        if (hasTOFCluster) {
          o2::track::TrackPar trackTmpOut(tracksTPC[iTrk].getParamOut());
          if (trackTmpOut.rotate(o2::math_utils::sector2Angle(tofCl.getSector())) && propagator->propagateTo(trackTmpOut, tofCl.getX(), false, mMaxSnp, mFineStep, mMatType)) {
            tpcYDeltaAtTOF = trackTmpOut.getY() - tofCl.getY();
            tpcZDeltaAtTOF = signSide * (o2::tpc::ParameterElectronics::Instance().ZbinWidth * trackFull.getTime0() - vertex.getTimeStamp().getTimeStamp()) * mVDrift - trackTmpOut.getZ() + tofCl.getZ();
          }
        }

        // get delta parameter between inner and outer
        float deltaTPCParamInOutTgl = trackFull.getTgl() - trackFull.getParamOut().getTgl();
        float deltaTPCParamInOutQPt = trackFull.getQ2Pt() - trackFull.getParamOut().getQ2Pt();

        *mStreamer[iThread] << "treeTimeSeries"
                            // DCAs
                            << "factorPt=" << factorPt
                            << "weight=" << weight
                            << "dcar_tpc=" << dca[0]
                            << "dcaz_tpc=" << dca[1]
                            << "dcar_itstpc=" << dcaITSTPC[0]
                            << "dcaz_itstpc=" << dcaITSTPC[1]
                            << "dcarW=" << dcarW
                            << "dcaZFromDeltaTime=" << dcaZFromDeltaTime
                            << "hasITSTPC=" << hasITSTPC
                            // vertex
                            << "vertex_x=" << vertex.getX()
                            << "vertex_y=" << vertex.getY()
                            << "vertex_z=" << vertex.getZ()
                            << "vertex_time=" << vertex.getTimeStamp().getTimeStamp()
                            << "vertex_nContributors=" << vertex.getNContributors()
                            << "isNearestVertex=" << isNearestVtx
                            // tpc track properties
                            << "pt=" << trkOrig.getPt()
                            << "qpt_ITSTPC=" << pt_ITS
                            << "tpc_timebin=" << trkOrig.getTime0()
                            << "qpt=" << trkOrig.getParam(4)
                            << "ncl=" << trkOrig.getNClusters()
                            << "tgl=" << trkOrig.getTgl()
                            << "side_type=" << typeSide
                            << "phi=" << trkOrig.getPhi()
                            << "clusterMask=" << clusterMask
                            << "dedxTPC=" << trkOrig.getdEdx()
                            << "chi2=" << trkOrig.getChi2()
                            << "mX=" << trkOrig.getX()
                            << "mX_ITS=" << mx_ITS
                            << "nClITS=" << nClITS
                            << "chi2ITS=" << chi2ITS
                            << "chi2match_ITSTPC=" << chi2match_ITSTPC
                            //
                            << "deltaPar2=" << deltaP2
                            << "deltaPar3=" << deltaP3
                            << "deltaPar4=" << deltaP4
                            << "sigmaY2=" << sigmaY2
                            << "sigmaZ2=" << sigmaZ2
                            // meta
                            << "mult=" << mNTracksWindow[iTrk]
                            << "time_window_mult=" << mTimeWindowMUS
                            << "firstTFOrbit=" << mFirstTFOrbit
                            << "mVDrift=" << mVDrift
                            << "its_flag=" << int(gID)
                            << "sqrtChi2Match=" << chi2Match
                            // TOF cluster
                            << "tpcYDeltaAtTOF=" << tpcYDeltaAtTOF
                            << "tpcZDeltaAtTOF=" << tpcZDeltaAtTOF
                            << "mDXatTOF=" << std::get<1>(idxTPCTrackToTOFCluster[iTrk])
                            << "mDZatTOF=" << std::get<2>(idxTPCTrackToTOFCluster[iTrk])
                            // TPC delta param
                            << "deltaTPCParamInOutTgl=" << deltaTPCParamInOutTgl
                            << "deltaTPCParamInOutQPt=" << deltaTPCParamInOutQPt
                            << "\n";
      }
    }
  }

  void sendOutput(ProcessingContext& pc)
  {
    pc.outputs().snapshot(Output{header::gDataOriginTPC, getDataDescriptionTimeSeries()}, mBufferDCA);
    // in case of ROOT output also store the TFinfo in the TTree
    if (!mDisableWriter) {
      o2::dataformats::TFIDInfo tfinfo;
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, tfinfo);
      const long timeMS = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS() + processing_helpers::getFirstTForbit(pc) * o2::constants::lhc::LHCOrbitMUS / 1000;
      mBufferDCA.mTSTPC.setStartTime(timeMS);
      mBufferDCA.mTSITSTPC.setStartTime(timeMS);
      pc.outputs().snapshot(Output{header::gDataOriginTPC, getDataDescriptionTPCTimeSeriesTFId()}, tfinfo);
    }
  }

  /// find for tpc tracks the nearest vertex
  void findNearesVertex(const gsl::span<const TrackTPC> tracksTPC, const gsl::span<const o2::dataformats::PrimaryVertex> vertices)
  {
    // create list of time bins of tracks
    const int nVertices = vertices.size();

    const int nTracks = tracksTPC.size();
    mNearestVtxTPC.clear();
    mNearestVtxTPC.resize(nTracks);

    // in case no vertices are found
    if (!nVertices) {
      std::fill(mNearestVtxTPC.begin(), mNearestVtxTPC.end(), -1);
      return;
    }

    // store timestamps of vertices. Assume vertices are already sorted in time!
    std::vector<float> times_vtx;
    times_vtx.reserve(nVertices);
    for (const auto& vtx : vertices) {
      times_vtx.emplace_back(vtx.getTimeStamp().getTimeStamp());
    }

    // loop over tpc tracks and find nearest vertex
    auto myThread = [&](int iThread) {
      for (int i = iThread; i < nTracks; i += mNThreads) {
        const float timeTrack = o2::tpc::ParameterElectronics::Instance().ZbinWidth * tracksTPC[i].getTime0();
        const auto lower = std::lower_bound(times_vtx.begin(), times_vtx.end(), timeTrack);
        int closestVtx = std::distance(times_vtx.begin(), lower);
        // if value is out of bounds use last value
        if (closestVtx == nVertices) {
          closestVtx -= 1;
        } else if (closestVtx > 0) {
          // if idx > 0 check preceeding value
          double diff1 = std::abs(timeTrack - *lower);
          double diff2 = std::abs(timeTrack - *(lower - 1));
          if (diff2 < diff1) {
            closestVtx -= 1;
          }
        }
        mNearestVtxTPC[i] = closestVtx;
      }
    };

    std::vector<std::thread> threads(mNThreads);
    for (int i = 0; i < mNThreads; i++) {
      threads[i] = std::thread(myThread, i);
    }

    // wait for the threads to finish
    for (auto& th : threads) {
      th.join();
    }
  }

  /// make bit mask of clusters
  std::vector<bool> makeClusterBitMask(const TrackTPC& track) const
  {
    std::vector<bool> tpcClusterMask(Mapper::PADROWS, false);
    const int nCl = track.getNClusterReferences();
    for (int j = 0; j < nCl; ++j) {
      uint8_t sector, padrow;
      uint32_t clusterIndexInRow;
      track.getClusterReference(mTPCTrackClIdx, j, sector, padrow, clusterIndexInRow);
      tpcClusterMask[padrow] = true;
    }
    return tpcClusterMask;
  }

  /// find number of neighbouring tracks for mult estimate
  void findNNeighbourTracks(const gsl::span<const TrackTPC> tracksTPC)
  {
    const float tpcTBinMUS = o2::tpc::ParameterElectronics::Instance().ZbinWidth; // 0.199606f; time bin in MUS
    const float windowTimeBins = mTimeWindowMUS / tpcTBinMUS;                     // number of neighbouring time bins to check

    // create list of time bins of tracks
    std::vector<float> times;
    const int nTracks = tracksTPC.size();
    times.reserve(nTracks);
    for (const auto& trk : tracksTPC) {
      times.emplace_back(trk.getTime0());
    }
    std::sort(times.begin(), times.end());

    mNTracksWindow.clear();
    mNTracksWindow.resize(nTracks);

    // loop over tpc tracks and count number of neighouring tracks
    auto myThread = [&](int iThread) {
      for (int i = iThread; i < nTracks; i += mNThreads) {
        const float t0 = tracksTPC[i].getTime0();
        const auto upperV0 = std::upper_bound(times.begin(), times.end(), t0 + windowTimeBins);
        const auto lowerV0 = std::lower_bound(times.begin(), times.end(), t0 - windowTimeBins);
        const int nMult = std::distance(times.begin(), upperV0) - std::distance(times.begin(), lowerV0);
        mNTracksWindow[i] = nMult;
      }
    };

    std::vector<std::thread> threads(mNThreads);
    for (int i = 0; i < mNThreads; i++) {
      threads[i] = std::thread(myThread, i);
    }

    // wait for the threads to finish
    for (auto& th : threads) {
      th.join();
    }
  }

  std::unordered_map<unsigned int, int> processVertices(const gsl::span<const o2::dataformats::PrimaryVertex> vertices, const gsl::span<const o2::dataformats::VtxTrackIndex> primMatchedTracks, const gsl::span<const o2::dataformats::VtxTrackRef> primMatchedTracksRef, const RecoContainer& recoData)
  {
    // storing collision vertex to ITS-TPC track index
    std::unordered_map<unsigned int, int> indicesITSTPC_vtx; // ITS-TPC track index -> collision vertex ID

    std::unordered_map<int, int> nContributors_ITS;    // ITS: vertex ID -> n contributors
    std::unordered_map<int, int> nContributors_ITSTPC; // ITS-TPC (and ITS-TPC-TRD, ITS-TPC-TOF, ITS-TPC-TRD-TOF): vertex ID -> n contributors

    // loop over collisions
    if (!vertices.empty()) {
      for (const auto& ref : primMatchedTracksRef) {
        // loop over ITS and ITS-TPC sources
        const std::array<TrkSrc, 5> sources = {TrkSrc::ITSTPC, TrkSrc::ITSTPCTRD, TrkSrc::ITSTPCTOF, TrkSrc::ITSTPCTRDTOF, TrkSrc::ITS};
        for (auto source : sources) {
          const int vID = ref.getVtxID(); // vertex ID
          const int firstEntry = ref.getFirstEntryOfSource(source);
          const int nEntries = ref.getEntriesOfSource(source);
          // loop over all tracks belonging to the vertex
          for (int i = 0; i < nEntries; ++i) {
            const auto& matchedTrk = primMatchedTracks[i + firstEntry];
            bool pvCont = matchedTrk.isPVContributor();
            if (pvCont) {
              // store index of ITS-TPC track container and vertex ID
              auto refITSTPC = recoData.getSingleDetectorRefs(matchedTrk)[TrkSrc::ITSTPC];
              if (refITSTPC.isIndexSet()) {
                indicesITSTPC_vtx[refITSTPC] = vID;
                ++nContributors_ITSTPC[vID];
              } else {
                ++nContributors_ITS[vID];
              }
            }
          }
        }
      }
    }

    // calculate statistics
    std::array<RobustAverage, 4> avgVtxITS;
    std::array<RobustAverage, 4> avgVtxITSTPC;
    for (int i = 0; i < avgVtxITS.size(); ++i) {
      avgVtxITS[i].reserve(vertices.size());
      avgVtxITSTPC[i].reserve(vertices.size());
    }
    for (int ivtx = 0; ivtx < vertices.size(); ++ivtx) {
      const auto& vtx = vertices[ivtx];
      const float itsFrac = nContributors_ITS[ivtx] / static_cast<float>(vtx.getNContributors());
      const float itstpcFrac = (nContributors_ITS[ivtx] + nContributors_ITSTPC[ivtx]) / static_cast<float>(vtx.getNContributors());

      const float itsMin = 0.2;
      const float itsMax = 0.8;
      if ((itsFrac > itsMin) && (itsFrac < itsMax)) {
        avgVtxITS[0].addValue(vtx.getX());
        avgVtxITS[1].addValue(vtx.getY());
        avgVtxITS[2].addValue(vtx.getZ());
        avgVtxITS[3].addValue(vtx.getNContributors());
      }

      const float itstpcMax = 0.95;
      if (itstpcFrac < itstpcMax) {
        avgVtxITSTPC[0].addValue(vtx.getX());
        avgVtxITSTPC[1].addValue(vtx.getY());
        avgVtxITSTPC[2].addValue(vtx.getZ());
        avgVtxITSTPC[3].addValue(vtx.getNContributors());
      }
    }

    // ITS
    mBufferDCA.nPrimVertices_ITS.front() = avgVtxITS[3].getValues().size();
    mBufferDCA.nVertexContributors_ITS_Median.front() = avgVtxITS[3].getMedian();
    mBufferDCA.nVertexContributors_ITS_RMS.front() = avgVtxITS[3].getStdDev();
    mBufferDCA.vertexX_ITS_Median.front() = avgVtxITS[0].getMedian();
    mBufferDCA.vertexY_ITS_Median.front() = avgVtxITS[1].getMedian();
    mBufferDCA.vertexZ_ITS_Median.front() = avgVtxITS[2].getMedian();
    mBufferDCA.vertexX_ITS_RMS.front() = avgVtxITS[0].getStdDev();
    mBufferDCA.vertexY_ITS_RMS.front() = avgVtxITS[1].getStdDev();
    mBufferDCA.vertexZ_ITS_RMS.front() = avgVtxITS[2].getStdDev();

    // ITS-TPC
    mBufferDCA.nPrimVertices_ITSTPC.front() = avgVtxITSTPC[3].getValues().size();
    mBufferDCA.nVertexContributors_ITSTPC_Median.front() = avgVtxITSTPC[3].getMedian();
    mBufferDCA.nVertexContributors_ITSTPC_RMS.front() = avgVtxITSTPC[3].getStdDev();
    mBufferDCA.vertexX_ITSTPC_Median.front() = avgVtxITSTPC[0].getMedian();
    mBufferDCA.vertexY_ITSTPC_Median.front() = avgVtxITSTPC[1].getMedian();
    mBufferDCA.vertexZ_ITSTPC_Median.front() = avgVtxITSTPC[2].getMedian();
    mBufferDCA.vertexX_ITSTPC_RMS.front() = avgVtxITSTPC[0].getStdDev();
    mBufferDCA.vertexY_ITSTPC_RMS.front() = avgVtxITSTPC[1].getStdDev();
    mBufferDCA.vertexZ_ITSTPC_RMS.front() = avgVtxITSTPC[2].getStdDev();

    // quantiles and truncated mean
    RobustAverage avg(vertices.size(), false);
    for (const auto& vtx : vertices) {
      if (vtx.getNContributors() > mMinTracksPerVertex) {
        // transform by n^0.5 to get more flat distribution
        avg.addValue(std::sqrt(vtx.getNContributors()));
      }
    }

    // nPrimVertices_Quantiles
    int sizeQ = mBufferDCA.nVertexContributors_Quantiles.size();
    const int nBinsQ = 20;
    if (sizeQ >= (nBinsQ + 3)) {
      for (int iq = 0; iq < nBinsQ; ++iq) {
        const float quantile = (iq + 1) / static_cast<float>(nBinsQ);
        const float val = avg.getQuantile(quantile, 1);
        mBufferDCA.nVertexContributors_Quantiles[iq] = val * val;
      }
      const float tr0 = avg.getTrunctedMean(0.05, 0.95);
      const float tr1 = avg.getTrunctedMean(0.1, 0.9);
      const float tr2 = avg.getTrunctedMean(0.2, 0.8);
      mBufferDCA.nVertexContributors_Quantiles[sizeQ - 3] = tr0 * tr0;
      mBufferDCA.nVertexContributors_Quantiles[sizeQ - 2] = tr1 * tr1;
      mBufferDCA.nVertexContributors_Quantiles[sizeQ - 1] = tr2 * tr2;
    }
    mBufferDCA.nPrimVertices.front() = vertices.size();

    return indicesITSTPC_vtx;
  }

  /// \return returns total number of stored values per TF
  int getNBins() const { return mBufferDCA.mTSTPC.getNBins(); }

  ValsdEdx getdEdxVars(bool useQMax, const TrackTPC& track) const
  {
    const float dedx = useQMax ? track.getdEdx().dEdxMaxTPC : track.getdEdx().dEdxTotTPC;
    const float dedxRatioNorm = mPID.getExpectedSignal(track, o2::track::PID::ID(o2::track::PID::Pion)) / dedx;
    float dedxNorm = (dedxRatioNorm > 0) ? std::log(mPID.getExpectedSignal(track, o2::track::PID::ID(o2::track::PID::Pion)) / dedx) : -1;
    // restrict to specified range
    if (std::abs(dedxNorm) > mMaxdEdxRatio) {
      dedxNorm = -1;
    }

    // get log(dedxRegion / dedx)
    float dedxIROC = -1;
    float dedxOROC1 = -1;
    float dedxOROC2 = -1;
    float dedxOROC3 = -1;
    if (dedx > 0) {
      dedxIROC = useQMax ? track.getdEdx().dEdxMaxIROC : track.getdEdx().dEdxTotIROC;
      dedxOROC1 = useQMax ? track.getdEdx().dEdxMaxOROC1 : track.getdEdx().dEdxTotOROC1;
      dedxOROC2 = useQMax ? track.getdEdx().dEdxMaxOROC2 : track.getdEdx().dEdxTotOROC2;
      dedxOROC3 = useQMax ? track.getdEdx().dEdxMaxOROC3 : track.getdEdx().dEdxTotOROC3;
      dedxIROC /= dedx;
      dedxOROC1 /= dedx;
      dedxOROC2 /= dedx;
      dedxOROC3 /= dedx;
      dedxIROC = (dedxIROC > 0) ? std::log(dedxIROC) : -1;
      dedxOROC1 = (dedxOROC1 > 0) ? std::log(dedxOROC1) : -1;
      dedxOROC2 = (dedxOROC2 > 0) ? std::log(dedxOROC2) : -1;
      dedxOROC3 = (dedxOROC3 > 0) ? std::log(dedxOROC3) : -1;

      // restrict to specified range
      if (std::abs(dedxIROC) > mMaxdEdxRegionRatio) {
        dedxIROC = -1;
      }
      if (std::abs(dedxOROC1) > mMaxdEdxRegionRatio) {
        dedxOROC1 = -1;
      }
      if (std::abs(dedxOROC2) > mMaxdEdxRegionRatio) {
        dedxOROC2 = -1;
      }
      if (std::abs(dedxOROC3) > mMaxdEdxRegionRatio) {
        dedxOROC3 = -1;
      }
    }
    return ValsdEdx{dedxNorm, dedxIROC, dedxOROC1, dedxOROC2, dedxOROC3};
  }
};

o2::framework::DataProcessorSpec getTPCTimeSeriesSpec(const bool disableWriter, const o2::base::Propagator::MatCorrType matType, const bool enableUnbinnedWriter, bool tpcOnly)
{
  using GID = o2::dataformats::GlobalTrackID;
  auto dataRequest = std::make_shared<DataRequest>();
  bool useMC = false;
  GID::mask_t srcTracks = tpcOnly ? GID::getSourcesMask("TPC") : GID::getSourcesMask("TPC,ITS,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  dataRequest->requestTracks(srcTracks, useMC);
  if (!tpcOnly) {
    dataRequest->requestPrimaryVertertices(useMC);
  }

  const bool enableAskMatLUT = matType == o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(!disableWriter,                 // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                true,                           // GRPMagField
                                                                enableAskMatLUT,                // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                dataRequest->inputs,
                                                                true,
                                                                true);

  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTimeSeries(), 0, Lifetime::Sporadic);
  if (!disableWriter) {
    outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTPCTimeSeriesTFId(), 0, Lifetime::Sporadic);
  }

  return DataProcessorSpec{
    "tpc-time-series",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCTimeSeries>(ccdbRequest, disableWriter, matType, enableUnbinnedWriter, tpcOnly, dataRequest)},
    Options{
      {"min-momentum", VariantType::Float, 0.2f, {"Minimum momentum of the tracks"}},
      {"min-cluster", VariantType::Int, 80, {"Minimum number of clusters of the tracks"}},
      {"max-tgl", VariantType::Float, 1.1f, {"Maximum accepted tgl of the tracks"}},
      {"max-qPt", VariantType::Float, 5.f, {"Maximum abs(qPt) bin"}},
      {"max-snp", VariantType::Float, 0.85f, {"Maximum sinus(phi) for propagation"}},
      {"coarse-step", VariantType::Float, 5.f, {"Coarse step during track propagation"}},
      {"fine-step", VariantType::Float, 2.f, {"Fine step during track propagation"}},
      {"mX-coarse", VariantType::Float, 40.f, {"Perform coarse propagation up to this mx"}},
      {"max-tracks", VariantType::Int, -1, {"Number of maximum tracks to process"}},
      {"cut-DCA-median", VariantType::Float, 3.f, {"Cut on the DCA: abs(DCA-medianDCA)<cut-DCA-median"}},
      {"cut-DCA-RMS", VariantType::Float, 3.f, {"Sigma cut on the DCA"}},
      {"refX-for-sector", VariantType::Float, 108.475f, {"Reference local x position for the sector information (default centre of IROC)"}},
      {"tgl-bins", VariantType::Int, 10, {"Number of tgl bins for time series variables"}},
      {"phi-bins", VariantType::Int, 18, {"Number of phi bins for time series variables"}},
      {"qPt-bins", VariantType::Int, 18, {"Number of qPt bins for time series variables"}},
      {"mult-bins", VariantType::Int, 20, {"Number of multiplicity bins for time series variables"}},
      {"mult-max", VariantType::Int, 80000, {"MAximum multiplicity bin"}},
      {"threads", VariantType::Int, 4, {"Number of parallel threads"}},
      {"max-ITS-TPC-DCAr", VariantType::Float, 0.2f, {"Maximum absolut DCAr value for ITS-TPC tracks"}},
      {"max-ITS-TPC-DCAz", VariantType::Float, 10.f, {"Maximum absolut DCAz value for ITS-TPC tracks - larger due to vertex spread"}},
      {"max-ITS-TPC-DCAr_comb", VariantType::Float, 0.2f, {"Maximum absolut DCAr value for ITS-TPC tracks to vertex for combined DCA"}},
      {"max-ITS-TPC-DCAz_comb", VariantType::Float, 0.2f, {"Maximum absolut DCAr value for ITS-TPC tracks to vertex for combined DCA"}},
      {"MIP-dedx", VariantType::Float, 50.f, {"MIP dEdx for MIP/dEdx monitoring"}},
      {"time-window-mult-mus", VariantType::Float, 50.f, {"Time window in micro s for multiplicity estimate"}},
      {"sqrts", VariantType::Float, 13600.f, {"Centre of mass energy used for downsampling"}},
      {"min-tracks-per-vertex", VariantType::Int, 6, {"Minimum number of tracks per vertex required"}},
      {"max-dedx-ratio", VariantType::Float, 0.3f, {"Maximum absolute log(dedx(pion)/dedx) ratio"}},
      {"max-dedx-region-ratio", VariantType::Float, 0.5f, {"Maximum absolute log(dedx(region)/dedx) ratio"}},
      {"sample-unbinned-tsallis", VariantType::Bool, false, {"Perform sampling of unbinned data based on Tsallis function"}},
      {"sampling-factor", VariantType::Float, 0.001f, {"Sampling factor in case sample-unbinned-tsallis is used"}},
      {"out-file-unbinned", VariantType::String, "time_series_tracks.root", {"name of the output file for the unbinned data"}}}};
}

} // namespace tpc
} // end namespace o2
