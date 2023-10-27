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
#include "CommonUtils/DebugStreamer.h"
#include "MathUtils/Tsallis.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "CommonDataFormat/AbstractRefAccessor.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCCalibration/VDriftHelper.h"
#include <random>
#include <chrono>

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCTimeSeries : public Task
{
 public:
  /// \constructor
  TPCTimeSeries(std::shared_ptr<o2::base::GRPGeomRequest> req, const bool disableWriter, const o2::base::Propagator::MatCorrType matType) : mCCDBRequest(req), mDisableWriter(disableWriter), mMatType(matType){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNMaxTracks = ic.options().get<int>("max-tracks");
    mMinMom = ic.options().get<float>("min-momentum");
    mMinNCl = ic.options().get<int>("min-cluster");
    mMaxTgl = ic.options().get<float>("max-tgl");
    mMaxQPtBin = ic.options().get<float>("max-qPt-bin");
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
    mUseQMax = ic.options().get<bool>("use-qMax");
    mMaxSnp = ic.options().get<float>("max-snp");
    mXCoarse = ic.options().get<float>("mX-coarse");
    mSqrt = ic.options().get<float>("sqrts");
    mBufferVals.resize(mNThreads);
    mBufferDCA.setBinning(mPhiBins, mTglBins, mQPtBins);
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
      mMIPdEdxRatioA.resize(nBins);
      mMIPdEdxRatioC.resize(nBins);
      mTPCChi2A.resize(nBins);
      mTPCChi2C.resize(nBins);
      mTPCNClA.resize(nBins);
      mTPCNClC.resize(nBins);
    }

    // getting tracks
    auto tracksTPC = pc.inputs().get<gsl::span<TrackTPC>>("tracksTPC");
    auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("tracksITSTPC");

    // getting the vertices
    const auto vertices = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx");
    const auto primMatchedTracks = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_trmtc");
    const auto primMatchedTracksRef = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_tref");

    LOGP(info, "Processing {} vertices, {} primary matched tracks, {} TPC tracks, {} ITS-TPC tracks", vertices.size(), primMatchedTracks.size(), tracksTPC.size(), tracksITSTPC.size());

    // calculate mean vertex, RMS and count vertices
    std::array<RobustAverage, 4> avgVtx;
    for (auto& avg : avgVtx) {
      avg.reserve(vertices.size());
    }
    for (const auto& vtx : vertices) {
      avgVtx[0].addValue(vtx.getX());
      avgVtx[1].addValue(vtx.getY());
      avgVtx[2].addValue(vtx.getZ());
      avgVtx[3].addValue(vtx.getNContributors());
    }
    mBufferDCA.vertexX_Median.front() = avgVtx[0].getMedian();
    mBufferDCA.vertexY_Median.front() = avgVtx[1].getMedian();
    mBufferDCA.vertexZ_Median.front() = avgVtx[2].getMedian();
    mBufferDCA.vertexX_RMS.front() = avgVtx[0].getStdDev();
    mBufferDCA.vertexY_RMS.front() = avgVtx[1].getStdDev();
    mBufferDCA.vertexZ_RMS.front() = avgVtx[2].getStdDev();
    mBufferDCA.nPrimVertices.front() = vertices.size();
    mBufferDCA.nVertexContributors_Median.front() = avgVtx[3].getMedian();
    mBufferDCA.nVertexContributors_RMS.front() = avgVtx[3].getStdDev();

    // storing collision vertex to ITS-TPC track index
    std::unordered_map<unsigned int, int> indicesITSTPC_vtx; // ITS-TPC track index -> collision vertex ID
    // loop over collisions
    for (const auto& ref : primMatchedTracksRef) {
      const auto source = o2::dataformats::VtxTrackIndex::Source::ITSTPC;
      const int vID = ref.getVtxID(); // vertex ID
      const int firstEntry = ref.getFirstEntryOfSource(source);
      const int nEntries = ref.getEntriesOfSource(source);
      // loop over all tracks belonging to the vertex
      for (int i = 0; i < nEntries; ++i) {
        // storing vertex ID for ITS-TPC track
        const auto& matchedTrk = primMatchedTracks[i + firstEntry];
        if (matchedTrk.isTrackSource(source)) {
          indicesITSTPC_vtx[matchedTrk] = vID;
        }
      }
    }

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

    // find nearest vertex of tracks which have no vertex assigned
    findNearesVertex(tracksTPC, vertices);

    // getting cluster references for cluster bitmask
    GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamTimeSeries)) {
      mTPCTrackClIdx = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
      mFirstTFOrbit = processing_helpers::getFirstTForbit(pc);
      // get local multiplicity - count neighbouring tracks
      findNNeighbourTracks(tracksTPC);
    })

    // reset buffers
    for (int i = 0; i < nBins; ++i) {
      for (int type = 0; type < mAvgADCAr[i].size(); ++type) {
        mAvgADCAr[i][type].clear();
        mAvgCDCAr[i][type].clear();
        mAvgADCAz[i][type].clear();
        mAvgCDCAz[i][type].clear();
      }
      for (int type = 0; type < mMIPdEdxRatioA[i].size(); ++type) {
        mMIPdEdxRatioA[i][type].clear();
        mMIPdEdxRatioC[i][type].clear();
        mTPCChi2A[i][type].clear();
        mTPCChi2C[i][type].clear();
        mTPCNClA[i][type].clear();
        mTPCNClC[i][type].clear();
      }
      for (int j = 0; j < mAvgMeffA[i].size(); ++j) {
        mAvgMeffA[i][j].clear();
        mAvgMeffC[i][j].clear();
        mAvgChi2MatchA[i][j].clear();
        mAvgChi2MatchC[i][j].clear();
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
      for (int type = 0; type < mAvgADCAr[i].size(); ++type) {
        mAvgADCAr[i][type].reserve(loopEnd);
        mAvgCDCAr[i][type].reserve(loopEnd);
        mAvgADCAz[i][type].reserve(loopEnd);
        mAvgCDCAz[i][type].reserve(loopEnd);
      }
      for (int type = 0; type < mMIPdEdxRatioA[i].size(); ++type) {
        mMIPdEdxRatioA[i][type].reserve(loopEnd);
        mMIPdEdxRatioC[i][type].reserve(loopEnd);
        mTPCChi2A[i][type].reserve(loopEnd);
        mTPCChi2C[i][type].reserve(loopEnd);
        mTPCNClA[i][type].reserve(loopEnd);
        mTPCNClC[i][type].reserve(loopEnd);
      }
      for (int j = 0; j < mAvgMeffA[i].size(); ++j) {
        mAvgMeffA[i][j].reserve(loopEnd);
        mAvgMeffC[i][j].reserve(loopEnd);
        mAvgChi2MatchA[i][j].reserve(loopEnd);
        mAvgChi2MatchC[i][j].reserve(loopEnd);
      }
    }
    for (int iThread = 0; iThread < mNThreads; ++iThread) {
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
            fillDCA(tracksTPC, tracksITSTPC, vertices, i, iThread, indicesITSTPC);
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
            fillDCA(tracksTPC, tracksITSTPC, vertices, i, iThread, indicesITSTPC);
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
          const auto dcar = val.dcar[i];
          const auto dcaz = val.dcaz[i];
          const auto dcarW = val.dcarW[i];
          const int binInt = nBins - 1;
          const bool fillCombDCA = ((type == 1) && (val.dcarcomb[i] != -1) && (val.dcazcomb[i] != -1));
          const std::array<int, 4> bins{tglBin, phiBin, qPtBin, binInt};
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
        const auto dcar = val.dcar[i];
        const auto dcaz = val.dcaz[i];
        const auto hasITS = val.hasITS[i];
        const auto chi2Match = val.chi2Match[i];
        const auto dedxRatio = val.dedxRatio[i];
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
        auto& mAvgmMIPdEdxRatio = isCSide ? mMIPdEdxRatioC : mMIPdEdxRatioA;
        auto& mAvgmTPCChi2 = isCSide ? mTPCChi2C : mTPCChi2A;
        auto& mAvgmTPCNCl = isCSide ? mTPCNClC : mTPCNClA;

        const std::array<int, 4> bins{tglBin, phiBin, qPtBin, binInt};
        // fill bins
        for (auto bin : bins) {
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
            if (dedxRatio > 0) {
              mAvgmMIPdEdxRatio[bin][0].addValue(dedxRatio);
            }
            mAvgmTPCChi2[bin][0].addValue(sqrtChi2TPC);
            mAvgmTPCNCl[bin][0].addValue(nClTPC);
            if (hasITS) {
              if (dedxRatio > 0) {
                mAvgmMIPdEdxRatio[bin][1].addValue(dedxRatio);
              }
              mAvgmTPCChi2[bin][1].addValue(sqrtChi2TPC);
              mAvgmTPCNCl[bin][1].addValue(nClTPC);
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

      for (int i = 0; i < mMIPdEdxRatioC[slice].size(); ++i) {
        auto& buff = (i == 0) ? mBufferDCA.mTSTPC : mBufferDCA.mTSITSTPC;
        buff.mMIPdEdxRatioC[slice] = mMIPdEdxRatioC[slice][i].getMean();
        buff.mMIPdEdxRatioA[slice] = mMIPdEdxRatioA[slice][i].getMean();
        buff.mTPCChi2C[slice] = mTPCChi2C[slice][i].getMean();
        buff.mTPCChi2A[slice] = mTPCChi2A[slice][i].getMean();
        buff.mTPCNClC[slice] = mTPCNClC[slice][i].getMean();
        buff.mTPCNClA[slice] = mTPCNClA[slice][i].getMean();
      }
    }

    auto stop = timer::now();
    std::chrono::duration<float> time = stop - startTotal;
    LOGP(detail, "DCA processing took {}", time.count());

    // send data
    sendOutput(pc);
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    o2::utils::DebugStreamer::instance()->flush();
    eos.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    mTPCVDriftHelper.accountCCDBInputs(matcher, obj);
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 private:
  /// buffer struct for multithreading
  struct FillVals {
    void reserve(int n, int type)
    {
      side.reserve(n);
      tglBin.reserve(n);
      phiBin.reserve(n);
      qPtBin.reserve(n);
      dcar.reserve(n);
      dcaz.reserve(n);
      dcarW.reserve(n);
      dedxRatio.reserve(n);
      sqrtChi2TPC.reserve(n);
      nClTPC.reserve(n);
      if (type == 1) {
        hasITS.reserve(n);
        chi2Match.reserve(n);
        gID.reserve(n);
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
      dcar.clear();
      dcaz.clear();
      dcarW.clear();
      hasITS.clear();
      chi2Match.clear();
      dedxRatio.clear();
      sqrtChi2TPC.clear();
      nClTPC.clear();
      gID.clear();
      dcarcomb.clear();
      dcazcomb.clear();
    }

    void emplace_back(Side sideTmp, int tglBinTmp, int phiBinTmp, int qPtBinTmp, float dcarTmp, float dcazTmp, float dcarWTmp, float dedxRatioTmp, float sqrtChi2TPCTmp, float nClTPCTmp, o2::dataformats::GlobalTrackID::Source gIDTmp = o2::dataformats::GlobalTrackID::Source::NSources, float chi2MatchTmp = -2, int hasITSTmp = -1)
    {
      side.emplace_back(sideTmp);
      tglBin.emplace_back(tglBinTmp);
      phiBin.emplace_back(phiBinTmp);
      qPtBin.emplace_back(qPtBinTmp);
      dcar.emplace_back(dcarTmp);
      dcaz.emplace_back(dcazTmp);
      dcarW.emplace_back(dcarWTmp);
      dedxRatio.emplace_back(dedxRatioTmp);
      sqrtChi2TPC.emplace_back(sqrtChi2TPCTmp);
      nClTPC.emplace_back(nClTPCTmp);
      if (chi2MatchTmp != -2) {
        chi2Match.emplace_back(chi2MatchTmp);
        hasITS.emplace_back(hasITSTmp);
        gID.emplace_back(gIDTmp);
      }
    }

    void emplace_back_ITSTPC(Side sideTmp, int tglBinTmp, int phiBinTmp, int qPtBinTmp, float dcarTmp, float dcazTmp, float dcarWTmp, float dedxRatioTmp, float sqrtChi2TPCTmp, float nClTPCTmp, float dcarCombTmp, float dcazCombTmp)
    {
      side.emplace_back(sideTmp);
      tglBin.emplace_back(tglBinTmp);
      phiBin.emplace_back(phiBinTmp);
      qPtBin.emplace_back(qPtBinTmp);
      dcar.emplace_back(dcarTmp);
      dcaz.emplace_back(dcazTmp);
      dcarW.emplace_back(dcarWTmp);
      dedxRatio.emplace_back(dedxRatioTmp);
      sqrtChi2TPC.emplace_back(sqrtChi2TPCTmp);
      nClTPC.emplace_back(nClTPCTmp);
      dcarcomb.emplace_back(dcarCombTmp);
      dcazcomb.emplace_back(dcazCombTmp);
    }

    std::vector<Side> side;
    std::vector<int> tglBin;
    std::vector<int> phiBin;
    std::vector<int> qPtBin;
    std::vector<float> dcar;
    std::vector<float> dcaz;
    std::vector<float> dcarW;
    std::vector<bool> hasITS;
    std::vector<float> chi2Match;
    std::vector<float> dedxRatio;
    std::vector<float> sqrtChi2TPC;
    std::vector<float> nClTPC;
    std::vector<float> dcarcomb;
    std::vector<float> dcazcomb;
    std::vector<o2::dataformats::GlobalTrackID::Source> gID;
  };
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;   ///< info for CCDB request
  const bool mDisableWriter{false};                         ///< flag if no ROOT output will be written
  o2::base::Propagator::MatCorrType mMatType;               ///< material for propagation
  int mPhiBins = SECTORSPERSIDE;                            ///< number of phi bins
  int mTglBins{3};                                          ///< number of tgl bins
  int mQPtBins{20};                                         ///< number of qPt bins
  TimeSeriesITSTPC mBufferDCA;                              ///< buffer for integrate DCAs
  std::vector<std::array<RobustAverage, 3>> mAvgADCAr;      ///< for averaging the DCAr for TPC and ITS-TPC tracks for A-side
  std::vector<std::array<RobustAverage, 3>> mAvgCDCAr;      ///< for averaging the DCAr for TPC and ITS-TPC tracks for C-side
  std::vector<std::array<RobustAverage, 3>> mAvgADCAz;      ///< for averaging the DCAz for TPC and ITS-TPC tracks for A-side
  std::vector<std::array<RobustAverage, 3>> mAvgCDCAz;      ///< for averaging the DCAz for TPC and ITS-TPC tracks for C-side
  std::vector<std::array<RobustAverage, 2>> mMIPdEdxRatioA; ///< for averaging MIP/dEdx
  std::vector<std::array<RobustAverage, 2>> mMIPdEdxRatioC; ///< for averaging MIP/dEdx
  std::vector<std::array<RobustAverage, 2>> mTPCChi2A;      ///< for averaging chi2 TPC A
  std::vector<std::array<RobustAverage, 2>> mTPCChi2C;      ///< for averaging chi2 TPC C
  std::vector<std::array<RobustAverage, 2>> mTPCNClA;       ///< for averaging number of cluster A
  std::vector<std::array<RobustAverage, 2>> mTPCNClC;       ///< for averaging number of cluster C
  std::vector<std::array<RobustAverage, 3>> mAvgMeffA;      ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 3>> mAvgMeffC;      ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 3>> mAvgChi2MatchA; ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  std::vector<std::array<RobustAverage, 3>> mAvgChi2MatchC; ///< for matching efficiency ITS-TPC standalone + afterburner, standalone, afterburner
  int mNMaxTracks{-1};                                      ///< maximum number of tracks to process
  float mMinMom{1};                                         ///< minimum accepted momentum
  int mMinNCl{80};                                          ///< minimum accepted number of clusters per track
  float mMaxTgl{1};                                         ///< maximum eta
  float mMaxQPtBin{5};                                      ///< max qPt bin
  float mCoarseStep{1};                                     ///< coarse step during track propagation
  float mFineStep{0.005};                                   ///< fine step during track propagation
  float mCutDCA{5};                                         ///< cut on the abs(DCA-median)
  float mCutRMS{5};                                         ///< sigma cut for mean,median calculation
  float mRefXSec{108.475};                                  ///< reference lx position for sector information (default centre of IROC)
  int mNThreads{1};                                         ///< number of parallel threads
  float maxITSTPCDCAr{0.2};                                 ///< maximum abs DCAr value for ITS-TPC tracks
  float maxITSTPCDCAz{10};                                  ///< maximum abs DCAz value for ITS-TPC tracks
  float maxITSTPCDCAr_comb{0.2};                            ///< max abs DCA for ITS-TPC DCA to vertex
  float maxITSTPCDCAz_comb{0.2};                            ///< max abs DCA for ITS-TPC DCA to vertex
  gsl::span<const TPCClRefElem> mTPCTrackClIdx{};           ///< cluster refs for debugging
  std::vector<std::array<FillVals, 2>> mBufferVals;         ///< buffer for multithreading
  uint32_t mFirstTFOrbit{0};                                ///< first TF orbit
  float mTimeWindowMUS{50};                                 ///< time window in mus for local mult estimate
  float mMIPdEdx{50};                                       ///< MIP dEdx position for MIP/dEdx monitoring
  std::vector<int> mNTracksWindow;                          ///< number of tracks in time window
  std::vector<int> mNearestVtxTPC;                          ///< nearest vertex for tpc tracks
  o2::tpc::VDriftHelper mTPCVDriftHelper{};                 ///< helper for v-drift
  float mVDrift{2.64};                                      ///< v drift in mus
  bool mUseQMax{false};                                     ///< use dedx qmax for MIP/dEdx monitoring
  float mMaxSnp{0.85};                                      ///< max sinus phi for propagation
  float mXCoarse{40};                                       ///< perform propagation with coarse steps up to this mx
  float mSqrt{13600};                                       ///< centre of mass energy

  /// check if track passes coarse cuts
  bool acceptTrack(const TrackTPC& track) const
  {
    if ((track.getP() < mMinMom) || (track.getNClusters() < mMinNCl) || std::abs(track.getTgl()) > mMaxTgl) {
      return false;
    }
    return true;
  }

  void fillDCA(const gsl::span<const TrackTPC> tracksTPC, const gsl::span<const o2::dataformats::TrackTPCITS> tracksITSTPC, const gsl::span<const o2::dataformats::PrimaryVertex> vertices, const int iTrk, const int iThread, const std::unordered_map<unsigned int, std::array<int, 2>>& indicesITSTPC)
  {
    TrackTPC track = tracksTPC[iTrk];

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

    TrackTPC trackTmp = tracksTPC[iTrk];

    // coarse propagation to centre of IROC for phi bin
    if (!propagator->propagateTo(trackTmp, mRefXSec, false, mMaxSnp, mCoarseStep, mMatType)) {
      return;
    }

    const int tglBin = mTglBins * std::abs(trackTmp.getTgl()) / mMaxTgl + mPhiBins;
    const int phiBin = mPhiBins * trackTmp.getPhi() / o2::constants::math::TwoPI;
    const int qPtBin = mPhiBins + mTglBins + mQPtBins * (trackTmp.getQ2Pt() + mMaxQPtBin) / (2 * mMaxQPtBin);
    const int nBins = getNBins();

    if ((phiBin < 0) || (phiBin > mPhiBins) || (tglBin < mPhiBins) || (tglBin > (mPhiBins + mTglBins)) || (qPtBin < (mPhiBins + mTglBins)) || (qPtBin > (mPhiBins + mTglBins + mQPtBins))) {
      return;
    }

    const int sector = o2::math_utils::angle2Sector(trackTmp.getPhiPos());
    if (sector < SECTORSPERSIDE) {
      // find possible ITS-TPC track and vertex index
      auto it = indicesITSTPC.find(iTrk);
      const auto idxITSTPC = (it != indicesITSTPC.end()) ? (it->second) : std::array<int, 2>{-1, -1};

      // get vertex (check if vertex ID is valid). In case no vertex is assigned return nearest vertex or else default vertex
      const auto vertex = (idxITSTPC.back() != -1) ? vertices[idxITSTPC.back()] : ((mNearestVtxTPC[iTrk] != -1) ? vertices[mNearestVtxTPC[iTrk]] : o2::dataformats::PrimaryVertex{});

      // calculate DCAz: (time TPC track - time vertex) * vDrift + sign_side * vertexZ
      const float signSide = track.hasCSideClustersOnly() ? -1 : 1; // invert sign for C-side
      const float dcaZFromDeltaTime = (vertex.getTimeStamp().getTimeStamp() == 0) ? 0 : (o2::tpc::ParameterElectronics::Instance().ZbinWidth * track.getTime0() - vertex.getTimeStamp().getTimeStamp()) * mVDrift + signSide * vertex.getZ();

      // for weight of DCA
      const float resCl = std::min(track.getNClusters(), static_cast<int>(Mapper::PADROWS)) / static_cast<float>(Mapper::PADROWS);

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
      const float dedx = mUseQMax ? track.getdEdx().dEdxMaxTPC : track.getdEdx().dEdxTotTPC;
      const float dedxRatio = (dedx > 0) ? (mMIPdEdx / dedx) : -1;
      const float sqrtChi2TPC = (track.getChi2() > 0) ? std::sqrt(track.getChi2()) : 0;
      const float nClTPC = track.getNClusters();

      if (track.hasCSideClustersOnly()) {
        mBufferVals[iThread].front().emplace_back(Side::C, tglBin, phiBin, qPtBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatio, sqrtChi2TPC, nClTPC, gID, chi2Match, hasITSTPC);
      } else if (track.hasASideClustersOnly()) {
        mBufferVals[iThread].front().emplace_back(Side::A, tglBin, phiBin, qPtBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatio, sqrtChi2TPC, nClTPC, gID, chi2Match, hasITSTPC);
      }

      // check if the track was assigned to ITS track
      o2::gpu::gpustd::array<float, 2> dcaITSTPC{0, 0};
      if (hasITSTPC) {
        // propagate ITS-TPC track to (0,0)
        auto trackITSTPCTmp = tracksITSTPC[idxITSTPC.front()];
        // fine propagation with Bz only
        if (propagator->PropagateToXBxByBz(trackITSTPCTmp, mXCoarse, mMaxSnp, mCoarseStep, mMatType) && propagator->propagateToDCA(refPoint, trackITSTPCTmp, propagator->getNominalBz(), mFineStep, mMatType, &dcaITSTPC)) {
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
              dcaITSTPCTmp[0] = -1;
              dcaITSTPCTmp[1] = -1;
            }

            if (track.hasCSideClustersOnly()) {
              mBufferVals[iThread].back().emplace_back_ITSTPC(Side::C, tglBin, phiBin, qPtBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatio, sqrtChi2TPC, nClTPC, dcaITSTPCTmp[0], dcaITSTPCTmp[1]);
            } else if (track.hasASideClustersOnly()) {
              mBufferVals[iThread].back().emplace_back_ITSTPC(Side::A, tglBin, phiBin, qPtBin, dca[0], dcaZFromDeltaTime, dcarW, dedxRatio, sqrtChi2TPC, nClTPC, dcaITSTPCTmp[0], dcaITSTPCTmp[1]);
            }
          }
        }
      }

      GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamTimeSeries)) {
        const auto sampling = o2::utils::DebugStreamer::getSamplingTypeFrequency(o2::utils::StreamFlags::streamTimeSeries);
        const float factorPt = sampling.second;
        bool writeData = true;
        float weight = 0;
        if (sampling.first == o2::utils::SamplingTypes::sampleTsallis) {
          writeData = o2::math_utils::Tsallis::downsampleTsallisCharged(tracksTPC[iTrk].getPt(), factorPt, mSqrt, weight, o2::utils::DebugStreamer::getRandom());
        }
        if (writeData) {
          auto clusterMask = makeClusterBitMask(track);
          const auto& trkOrig = tracksTPC[iTrk];
          const bool isNearestVtx = (idxITSTPC.back() == -1); // is nearest vertex in case no vertex was found
          const float mx_ITS = hasITSTPC ? tracksITSTPC[idxITSTPC.front()].getX() : -1;
          int typeSide = 2; // A- and C-Side cluster
          if (track.hasASideClustersOnly()) {
            typeSide = 0;
          } else if (track.hasCSideClustersOnly()) {
            typeSide = 1;
          }

          o2::utils::DebugStreamer::instance()->getStreamer("time_series", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("treeTimeSeries").data()
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
                                                                                     << "tpc_timebin=" << trkOrig.getTime0()
                                                                                     << "qpt=" << trkOrig.getParam(4)
                                                                                     << "ncl=" << trkOrig.getNClusters()
                                                                                     << "tgl=" << trkOrig.getTgl()
                                                                                     << "side_type=" << typeSide
                                                                                     << "phi=" << trkOrig.getPhi()
                                                                                     << "clusterMask=" << clusterMask
                                                                                     << "dedxTotTPC=" << trkOrig.getdEdx().dEdxTotTPC
                                                                                     << "dedxTotIROC=" << trkOrig.getdEdx().dEdxTotIROC
                                                                                     << "dedxTotOROC1=" << trkOrig.getdEdx().dEdxTotOROC1
                                                                                     << "dedxTotOROC2=" << trkOrig.getdEdx().dEdxTotOROC2
                                                                                     << "dedxTotOROC3=" << trkOrig.getdEdx().dEdxTotOROC3
                                                                                     << "chi2=" << trkOrig.getChi2()
                                                                                     << "mX=" << trkOrig.getX()
                                                                                     << "mX_ITS=" << mx_ITS
                                                                                     // meta
                                                                                     << "mult=" << mNTracksWindow[iTrk]
                                                                                     << "time_window_mult=" << mTimeWindowMUS
                                                                                     << "firstTFOrbit=" << mFirstTFOrbit
                                                                                     << "mVDrift=" << mVDrift
                                                                                     << "\n";
        }
      })
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

  /// \return returns total number of stored values per TF
  int getNBins() const { return mBufferDCA.mTSTPC.getNBins(); }
};

o2::framework::DataProcessorSpec getTPCTimeSeriesSpec(const bool disableWriter, const o2::base::Propagator::MatCorrType matType)
{
  const bool enableAskMatLUT = matType == o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tracksITSTPC", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksTPC", header::gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe);
  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamTimeSeries)) {
    // request tpc clusters only in case the debug streamer is used for the cluster bit mask
    inputs.emplace_back("trackTPCClRefs", header::gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe);
  })
  inputs.emplace_back("pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe);
  inputs.emplace_back("pvtx_trmtc", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);    // global ids of associated tracks
  inputs.emplace_back("pvtx_tref", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe); // vertex - trackID refs

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(!disableWriter,                 // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                true,                           // GRPMagField
                                                                enableAskMatLUT,                // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs,
                                                                true,
                                                                true);

  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTimeSeries(), 0, Lifetime::Sporadic);
  if (!disableWriter) {
    outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTPCTimeSeriesTFId(), 0, Lifetime::Sporadic);
  }

  return DataProcessorSpec{
    "tpc-time-series",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCTimeSeries>(ccdbRequest, disableWriter, matType)},
    Options{{"min-momentum", VariantType::Float, 0.2f, {"Minimum momentum of the tracks"}},
            {"min-cluster", VariantType::Int, 80, {"Minimum number of clusters of the tracks"}},
            {"max-tgl", VariantType::Float, 1.1f, {"Maximum accepted tgl of the tracks"}},
            {"max-qPt-bin", VariantType::Float, 5.f, {"Maximum abs(qPt) bin"}},
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
            {"threads", VariantType::Int, 4, {"Number of parallel threads"}},
            {"max-ITS-TPC-DCAr", VariantType::Float, 0.2f, {"Maximum absolut DCAr value for ITS-TPC tracks"}},
            {"max-ITS-TPC-DCAz", VariantType::Float, 10.f, {"Maximum absolut DCAz value for ITS-TPC tracks - larger due to vertex spread"}},
            {"max-ITS-TPC-DCAr_comb", VariantType::Float, 0.2f, {"Maximum absolut DCAr value for ITS-TPC tracks to vertex for combined DCA"}},
            {"max-ITS-TPC-DCAz_comb", VariantType::Float, 0.2f, {"Maximum absolut DCAr value for ITS-TPC tracks to vertex for combined DCA"}},
            {"MIP-dedx", VariantType::Float, 50.f, {"MIP dEdx for MIP/dEdx monitoring"}},
            {"time-window-mult-mus", VariantType::Float, 50.f, {"Time window in micro s for multiplicity estimate"}},
            {"use-qMax", VariantType::Bool, false, {"Using qMax for MIP/dedx monitoring"}},
            {"sqrts", VariantType::Float, 13600.f, {"Centre of mass energy used for downsampling"}}}};
}

} // namespace tpc
} // end namespace o2
