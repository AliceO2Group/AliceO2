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

/// @file TracksToRecords.cxx

#include <iostream>
#include <sstream>
#include <string>

#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include <Framework/InputRecord.h>
#include "MFTBase/Geometry.h"
#include "MFTAlignment/MillePedeRecord.h"

#include "MFTAlignment/TracksToRecords.h"

using namespace o2::mft;

ClassImp(o2::mft::TracksToRecords);

//__________________________________________________________________________
TracksToRecords::TracksToRecords()
  : Aligner(),
    mRunNumber(0),
    mBz(0),
    mNumberTFs(0),
    mNumberOfClusterChainROFs(0),
    mNumberOfTrackChainROFs(0),
    mCounterLocalEquationFailed(0),
    mCounterSkippedTracks(0),
    mCounterUsedTracks(0),
    mGlobalDerivatives(std::vector<double>(mNumberOfGlobalParam)),
    mLocalDerivatives(std::vector<double>(mNumberOfTrackParam)),
    mMinNumberClusterCut(6),
    mWeightRecord(1.),
    mDictionary(nullptr),
    mAlignPoint(new AlignPointHelper()),
    mWithControl(false),
    mNEntriesAutoSave(10000),
    mRecordWriter(new MilleRecordWriter()),
    mWithConstraintsRecWriter(false),
    mConstraintsRecWriter(nullptr),
    mMillepede(new MillePede2())
{
  if (mWithConstraintsRecWriter) {
    mConstraintsRecWriter = new MilleRecordWriter();
  }
  // initialise the content of each array
  resetGlocalDerivative();
  resetLocalDerivative();
  LOGF(debug, "TracksToRecords instantiated");
}

//__________________________________________________________________________
TracksToRecords::~TracksToRecords()
{
  if (mConstraintsRecWriter) {
    delete mConstraintsRecWriter;
  }
  if (mMillepede) {
    delete mMillepede;
  }
  if (mRecordWriter) {
    delete mRecordWriter;
  }
  if (mAlignPoint) {
    delete mAlignPoint;
  }
  if (mDictionary)
    mDictionary = nullptr;
  LOGF(debug, "TracksToRecords destroyed");
}

//__________________________________________________________________________
void TracksToRecords::init()
{
  if (mIsInitDone)
    return;
  if (mDictionary == nullptr) {
    LOGF(fatal, "TracksToRecords::init() failed because no cluster dictionary is defined");
    mIsInitDone = false;
    return;
  }

  mRecordWriter->setCyclicAutoSave(mNEntriesAutoSave);
  mRecordWriter->setDataFileName(mMilleRecordsFileName);
  mMillepede->SetRecordWriter(mRecordWriter);

  if (mWithConstraintsRecWriter) {
    mConstraintsRecWriter->setCyclicAutoSave(mNEntriesAutoSave);
    mConstraintsRecWriter->setDataFileName(mMilleConstraintsRecFileName);
    mMillepede->SetConstraintsRecWriter(mConstraintsRecWriter);
  }

  mAlignPoint->setClusterDictionary(mDictionary);

  mMillepede->InitMille(mNumberOfGlobalParam,
                        mNumberOfTrackParam,
                        mChi2CutNStdDev,
                        mResCut,
                        mResCutInitial);

  LOG(info) << "-------------- TracksToRecords configured with -----------------";
  LOGF(info, "Chi2CutNStdDev = %d", mChi2CutNStdDev);
  LOGF(info, "ResidualCutInitial = %.3f", mResCutInitial);
  LOGF(info, "ResidualCut = %.3f", mResCut);
  LOGF(info, "MinNumberClusterCut = %d", mMinNumberClusterCut);
  LOGF(info, "mStartFac = %.3f", mStartFac);
  LOGF(info,
       "Allowed variation: dx = %.3f, dy = %.3f, dz = %.3f, dRz = %.4f",
       mAllowVar[0], mAllowVar[1], mAllowVar[3], mAllowVar[2]);
  LOG(info) << "-----------------------------------------------------------";

  // set allowed variations for all parameters
  for (int chipId = 0; chipId < mNumberOfSensors; ++chipId) {
    for (Int_t iPar = 0; iPar < mNDofPerSensor; ++iPar) {
      mMillepede->SetParSigma(chipId * mNDofPerSensor + iPar, mAllowVar[iPar]);
    }
  }

  // set iterations
  if (mStartFac > 1) {
    mMillepede->SetIterations(mStartFac);
  }

  mIsInitDone = true;
  LOGF(info, "TracksToRecords init done");
}

//__________________________________________________________________________
void TracksToRecords::processTimeFrame(o2::framework::ProcessingContext& ctx)
{
  mNumberTFs++; // TF Counter

  // get tracks
  mMFTTracks = ctx.inputs().get<gsl::span<o2::mft::TrackMFT>>("tracks");
  mMFTTracksROF = ctx.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("tracksrofs");
  mMFTTrackClusIdx = ctx.inputs().get<gsl::span<int>>("trackClIdx");

  // get clusters
  mMFTClusters = ctx.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  mMFTClustersROF = ctx.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clustersrofs");
  mMFTClusterPatterns = ctx.inputs().get<gsl::span<unsigned char>>("patterns");
  mPattIt = mMFTClusterPatterns.begin();
  mAlignPoint->convertCompactClusters(
    mMFTClusters, mPattIt, mMFTClustersLocal, mMFTClustersGlobal);
}

//__________________________________________________________________________
void TracksToRecords::processRecoTracks()
{
  if (!mIsInitDone) {
    LOGF(fatal, "TracksToRecords::processRecoTracks() aborted because init was not done !");
    return;
  }
  if (!mRecordWriter || !mRecordWriter->isInitOk()) {
    LOGF(fatal, "TracksToRecords::processRecoTracks() aborted because uninitialised mRecordWriter !");
    return;
  }

  LOG(info) << "TracksToRecords::processRecoTracks() - start";

  int nCounterAllTracks = 0;

  for (auto& oneTrack : mMFTTracks) { // track loop

    LOGF(debug, "Processing track # %5d", nCounterAllTracks);

    // Skip the track if not enough clusters
    auto ncls = oneTrack.getNumberOfPoints();
    if (ncls < mMinNumberClusterCut) {
      nCounterAllTracks++;
      mCounterSkippedTracks++;
      continue;
    }

    // Skip presumably quite low momentum track
    if (!oneTrack.isLTF()) {
      nCounterAllTracks++;
      mCounterSkippedTracks++;
      continue;
    }

    auto offset = oneTrack.getExternalClusterIndexOffset();

    mRecordWriter->getRecord()->Reset();

    // Store the initial track parameters
    auto track = oneTrack;
    mAlignPoint->resetTrackInitialParam();
    mAlignPoint->recordTrackInitialParam(track);

    bool isTrackUsed = true;

    for (int icls = 0; icls < ncls; ++icls) { // cluster loop

      mAlignPoint->resetAlignPoint();

      // Store measured positions
      auto clsEntry = mMFTTrackClusIdx[offset + icls];
      auto localCluster = mMFTClustersLocal[clsEntry];
      auto globalCluster = mMFTClustersGlobal[clsEntry];
      mAlignPoint->setMeasuredPosition(localCluster, globalCluster);
      if (!mAlignPoint->isClusterOk()) {
        LOGF(warning, "TracksToRecords::processRecoTracks() - will not use track # %5d with at least a bad cluster", nCounterAllTracks);
        mCounterSkippedTracks++;
        isTrackUsed = false;
        break;
      }

      // Propagate track to the current z plane of this cluster
      track.propagateParamToZlinear(mAlignPoint->getGlobalMeasuredPosition().Z());

      // Store reco positions
      mAlignPoint->setGlobalRecoPosition(track);

      // compute residuals
      mAlignPoint->setLocalResidual();
      mAlignPoint->setGlobalResidual();

      // Compute derivatives
      mAlignPoint->computeLocalDerivatives();
      mAlignPoint->computeGlobalDerivatives();

      // Set local equations
      bool success = true;
      success &= setLocalEquationX();
      success &= setLocalEquationY();
      success &= setLocalEquationZ();
      isTrackUsed &= success;
      if (mWithControl && success)
        mPointControl.fill(mAlignPoint, mCounterUsedTracks);
      if (!success) {
        LOGF(error, "TracksToRecords::processRecoTracks() - track %i h %d d %d l %d s %4d lMpos x %.2e y %.2e z %.2e gMpos x %.2e y %.2e z %.2e gRpos x %.2e y %.2e z %.2e",
             mCounterUsedTracks, mAlignPoint->half(), mAlignPoint->disk(), mAlignPoint->layer(), mAlignPoint->getSensorId(),
             mAlignPoint->getLocalMeasuredPosition().X(), mAlignPoint->getLocalMeasuredPosition().Y(), mAlignPoint->getLocalMeasuredPosition().Z(),
             mAlignPoint->getGlobalMeasuredPosition().X(), mAlignPoint->getGlobalMeasuredPosition().Y(), mAlignPoint->getGlobalMeasuredPosition().Z(),
             mAlignPoint->getGlobalRecoPosition().X(), mAlignPoint->getGlobalRecoPosition().Y(), mAlignPoint->getGlobalRecoPosition().Z());
      }

    } // end of loop on clusters

    if (isTrackUsed) {
      mRecordWriter->setRecordRun(mRunNumber);
      mRecordWriter->setRecordWeight(mWeightRecord);
      const bool doPrint = false;
      mRecordWriter->fillRecordTree(doPrint); // save record data
      mCounterUsedTracks++;
    }
    nCounterAllTracks++;
  } // end of loop on tracks
  LOG(info) << "TracksToRecords::processRecoTracks() - end";
}

//__________________________________________________________________________
void TracksToRecords::processROFs(TChain* mfttrackChain, TChain* mftclusterChain)
{
  if (!mIsInitDone) {
    LOGF(fatal, "TracksToRecords::processROFs() aborted because init was not done !");
    return;
  }

  if (!mRecordWriter || !mRecordWriter->isInitOk()) {
    LOGF(fatal, "TracksToRecords::processROFs() aborted because uninitialised mRecordWriter !");
    return;
  }

  LOG(info) << "TracksToRecords::processROFs() - start";

  TTreeReader mftTrackChainReader(mfttrackChain);
  TTreeReader mftClusterChainReader(mftclusterChain);
  std::vector<unsigned char>::iterator pattIterator;

  TTreeReaderValue<std::vector<o2::mft::TrackMFT>> mftTracks =
    {mftTrackChainReader, "MFTTrack"};
  TTreeReaderValue<std::vector<o2::itsmft::ROFRecord>> mftTracksROF =
    {mftTrackChainReader, "MFTTracksROF"};
  TTreeReaderValue<std::vector<int>> mftTrackClusIdx =
    {mftTrackChainReader, "MFTTrackClusIdx"};

  TTreeReaderValue<std::vector<o2::itsmft::CompClusterExt>> mftClusters =
    {mftClusterChainReader, "MFTClusterComp"};
  TTreeReaderValue<std::vector<o2::itsmft::ROFRecord>> mftClustersROF =
    {mftClusterChainReader, "MFTClustersROF"};
  TTreeReaderValue<std::vector<unsigned char>> mftClusterPatterns =
    {mftClusterChainReader, "MFTClusterPatt"};

  int nCounterAllTracks = 0;

  while (mftTrackChainReader.Next() && mftClusterChainReader.Next()) {

    mNumberOfTrackChainROFs += (*mftTracksROF).size();
    mNumberOfClusterChainROFs += (*mftClustersROF).size();
    assert(mNumberOfTrackChainROFs == mNumberOfClusterChainROFs);

    pattIterator = (*mftClusterPatterns).begin();
    mAlignPoint->convertCompactClusters(
      *mftClusters, pattIterator, mMFTClustersLocal, mMFTClustersGlobal);

    //______________________________________________________
    for (auto& oneTrack : *mftTracks) { // track loop

      LOGF(debug, "Processing track # %5d", nCounterAllTracks);

      // Skip the track if not enough clusters
      auto ncls = oneTrack.getNumberOfPoints();
      if (ncls < mMinNumberClusterCut) {
        nCounterAllTracks++;
        mCounterSkippedTracks++;
        continue;
      }

      // Skip presumably quite low momentum track
      if (!oneTrack.isLTF()) {
        nCounterAllTracks++;
        mCounterSkippedTracks++;
        continue;
      }

      auto offset = oneTrack.getExternalClusterIndexOffset();

      mRecordWriter->getRecord()->Reset();

      // Store the initial track parameters
      mAlignPoint->resetTrackInitialParam();
      mAlignPoint->recordTrackInitialParam(oneTrack);

      bool isTrackUsed = true;

      for (int icls = 0; icls < ncls; ++icls) { // cluster loop

        mAlignPoint->resetAlignPoint();

        // Store measured positions
        auto clsEntry = (*mftTrackClusIdx)[offset + icls];
        auto localCluster = mMFTClustersLocal[clsEntry];
        auto globalCluster = mMFTClustersGlobal[clsEntry];
        mAlignPoint->setMeasuredPosition(localCluster, globalCluster);
        if (!mAlignPoint->isClusterOk()) {
          LOGF(warning, "TracksToRecords::processROFs() - will not use track # %5d with at least a bad cluster", nCounterAllTracks);
          mCounterSkippedTracks++;
          isTrackUsed = false;
          break;
        }

        // Propagate track to the current z plane of this cluster
        oneTrack.propagateParamToZlinear(mAlignPoint->getGlobalMeasuredPosition().Z());

        // Store reco positions
        mAlignPoint->setGlobalRecoPosition(oneTrack);

        // compute residuals
        mAlignPoint->setLocalResidual();
        mAlignPoint->setGlobalResidual();

        // Compute derivatives
        mAlignPoint->computeLocalDerivatives();
        mAlignPoint->computeGlobalDerivatives();

        // Set local equations
        bool success = true;
        success &= setLocalEquationX();
        success &= setLocalEquationY();
        success &= setLocalEquationZ();
        isTrackUsed &= success;
        if (mWithControl && success)
          mPointControl.fill(mAlignPoint, mCounterUsedTracks);
        if (!success) {
          LOGF(error, "TracksToRecords::processROFs() - track %i h %d d %d l %d s %4d lMpos x %.2e y %.2e z %.2e gMpos x %.2e y %.2e z %.2e gRpos x %.2e y %.2e z %.2e",
               mCounterUsedTracks, mAlignPoint->half(), mAlignPoint->disk(), mAlignPoint->layer(), mAlignPoint->getSensorId(),
               mAlignPoint->getLocalMeasuredPosition().X(), mAlignPoint->getLocalMeasuredPosition().Y(), mAlignPoint->getLocalMeasuredPosition().Z(),
               mAlignPoint->getGlobalMeasuredPosition().X(), mAlignPoint->getGlobalMeasuredPosition().Y(), mAlignPoint->getGlobalMeasuredPosition().Z(),
               mAlignPoint->getGlobalRecoPosition().X(), mAlignPoint->getGlobalRecoPosition().Y(), mAlignPoint->getGlobalRecoPosition().Z());
        }

      } // end of loop on clusters

      if (isTrackUsed) {
        // copy track record
        mRecordWriter->setRecordRun(mRunNumber);
        mRecordWriter->setRecordWeight(mWeightRecord);
        const bool doPrint = false;
        mRecordWriter->fillRecordTree(doPrint); // save record data
        mCounterUsedTracks++;
      }
      nCounterAllTracks++;
    } // end of loop on tracks

  } // end of loop on TChain reader

  LOG(info) << "TracksToRecords::processROFs() - end";
}

//__________________________________________________________________________
void TracksToRecords::printProcessTrackSummary()
{
  LOGF(info, "TracksToRecords processRecoTracks() summary: ");
  if (mNumberOfTrackChainROFs) {
    LOGF(info,
         "n ROFs = %d, used tracks = %d, skipped tracks = %d, local equations failed = %d",
         mNumberOfTrackChainROFs, mCounterUsedTracks,
         mCounterSkippedTracks, mCounterLocalEquationFailed);
  } else {
    LOGF(info,
         "n TFs = %d, used tracks = %d, skipped tracks = %d, local equations failed = %d",
         mNumberTFs, mCounterUsedTracks,
         mCounterSkippedTracks, mCounterLocalEquationFailed);
  }
}

//__________________________________________________________________________
void TracksToRecords::startRecordWriter()
{
  if (mRecordWriter)
    mRecordWriter->init();
  if (mWithControl) {
    mPointControl.setCyclicAutoSave(mNEntriesAutoSave);
    mPointControl.init();
  }
}

//__________________________________________________________________________
void TracksToRecords::endRecordWriter()
{
  if (mRecordWriter) {
    mRecordWriter->terminate(); // write record tree and close output file
  }
  if (mWithControl)
    mPointControl.terminate();
}

//__________________________________________________________________________
void TracksToRecords::startConstraintsRecWriter()
{
  if (!mWithConstraintsRecWriter)
    return;

  if (mConstraintsRecWriter) {
    mConstraintsRecWriter->changeDataBranchName();
    mConstraintsRecWriter->init();
  }
}

//__________________________________________________________________________
void TracksToRecords::endConstraintsRecWriter()
{
  if (!mWithConstraintsRecWriter)
    return;

  if (mConstraintsRecWriter) {
    mConstraintsRecWriter->terminate();
  }
}

//__________________________________________________________________________
bool TracksToRecords::setLocalDerivative(Int_t index, Double_t value)
{
  // index [0 .. 3] for {dX0, dTx, dY0, dTz}

  bool success = false;
  if (index < mNumberOfTrackParam) {
    mLocalDerivatives[index] = value;
    success = true;
  } else {
    LOGF(error,
         "AlignHelper::setLocalDerivative() - index %d >= %d",
         index, mNumberOfTrackParam);
  }
  return success;
}

//__________________________________________________________________________
bool TracksToRecords::setGlobalDerivative(Int_t index, Double_t value)
{
  // index [0 .. 3] for {dDeltaX, dDeltaY, dDeltaRz, dDeltaZ}

  bool success = false;
  if (index < mNumberOfGlobalParam) {
    mGlobalDerivatives[index] = value;
    success = true;
  } else {
    LOGF(error,
         "AlignHelper::setGlobalDerivative() - index %d >= %d",
         index, mNumberOfGlobalParam);
  }
  return success;
}

//__________________________________________________________________________
bool TracksToRecords::resetLocalDerivative()
{
  bool success = false;
  std::fill(mLocalDerivatives.begin(), mLocalDerivatives.end(), 0.);
  success = true;
  return success;
}

//__________________________________________________________________________
bool TracksToRecords::resetGlocalDerivative()
{
  bool success = false;
  std::fill(mGlobalDerivatives.begin(), mGlobalDerivatives.end(), 0.);
  success = true;
  return success;
}

//__________________________________________________________________________
bool TracksToRecords::setLocalEquationX()
{

  if (!mAlignPoint->isAlignPointSet()) {
    LOGF(error,
         "TracksToRecords::setLocalEquationX() - no align point coordinates set !");
    return false;
  }
  if (!mAlignPoint->isGlobalDerivativeDone())
    return false;
  if (!mAlignPoint->isLocalDerivativeDone())
    return false;

  bool success = true;

  // clean slate for the local equation for this measurement

  success &= resetGlocalDerivative();
  success &= resetLocalDerivative();

  // local derivatives
  // index [0 .. 3] for {dX0, dTx, dY0, dTz}

  success &= setLocalDerivative(0, mAlignPoint->localDerivativeX().dX0());
  success &= setLocalDerivative(1, mAlignPoint->localDerivativeX().dTx());
  success &= setLocalDerivative(2, mAlignPoint->localDerivativeX().dY0());
  success &= setLocalDerivative(3, mAlignPoint->localDerivativeX().dTy());

  // global derivatives
  // index [0 .. 3] for {dDeltaX, dDeltaY, dDeltaRz, dDeltaZ}

  Int_t chipId = mAlignPoint->getSensorId();
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 0, mAlignPoint->globalDerivativeX().dDeltaX());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 1, mAlignPoint->globalDerivativeX().dDeltaY());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 2, mAlignPoint->globalDerivativeX().dDeltaRz());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 3, mAlignPoint->globalDerivativeX().dDeltaZ());

  if (success) {
    if (mCounterUsedTracks < 5) {
      LOGF(debug,
           "TracksToRecords::setLocalEquationX(): track %i sr %4d local %.3e %.3e %.3e %.3e, global %.3e %.3e %.3e %.3e X %.3e sigma %.3e",
           mCounterUsedTracks, chipId,
           mLocalDerivatives[0], mLocalDerivatives[1], mLocalDerivatives[2], mLocalDerivatives[3],
           mGlobalDerivatives[chipId * mNDofPerSensor + 0],
           mGlobalDerivatives[chipId * mNDofPerSensor + 1],
           mGlobalDerivatives[chipId * mNDofPerSensor + 2],
           mGlobalDerivatives[chipId * mNDofPerSensor + 3],
           mAlignPoint->getLocalMeasuredPosition().X(),
           mAlignPoint->getLocalMeasuredPositionSigma().X());
    }
    mMillepede->SetLocalEquation(
      mGlobalDerivatives,
      mLocalDerivatives,
      mAlignPoint->getLocalMeasuredPosition().X(),
      mAlignPoint->getLocalMeasuredPositionSigma().X());
  } else {
    mCounterLocalEquationFailed++;
  }

  return success;
}

//__________________________________________________________________________
bool TracksToRecords::setLocalEquationY()
{
  if (!mAlignPoint->isAlignPointSet()) {
    LOGF(error,
         "TracksToRecords::setLocalEquationY() - no align point coordinates set !");
    return false;
  }
  if (!mAlignPoint->isGlobalDerivativeDone())
    return false;
  if (!mAlignPoint->isLocalDerivativeDone())
    return false;

  bool success = true;

  // clean slate for the local equation for this measurement

  success &= resetGlocalDerivative();
  success &= resetLocalDerivative();

  // local derivatives
  // index [0 .. 3] for {dX0, dTx, dY0, dTz}

  success &= setLocalDerivative(0, mAlignPoint->localDerivativeY().dX0());
  success &= setLocalDerivative(1, mAlignPoint->localDerivativeY().dTx());
  success &= setLocalDerivative(2, mAlignPoint->localDerivativeY().dY0());
  success &= setLocalDerivative(3, mAlignPoint->localDerivativeY().dTy());

  // global derivatives
  // index [0 .. 3] for {dDeltaX, dDeltaY, dDeltaRz, dDeltaZ}

  Int_t chipId = mAlignPoint->getSensorId();
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 0, mAlignPoint->globalDerivativeY().dDeltaX());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 1, mAlignPoint->globalDerivativeY().dDeltaY());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 2, mAlignPoint->globalDerivativeY().dDeltaRz());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 3, mAlignPoint->globalDerivativeY().dDeltaZ());

  if (success) {
    if (mCounterUsedTracks < 5) {
      LOGF(debug,
           "TracksToRecords::setLocalEquationY(): track %i sr %4d local %.3e %.3e %.3e %.3e, global %.3e %.3e %.3e %.3e Y %.3e sigma %.3e",
           mCounterUsedTracks, chipId,
           mLocalDerivatives[0], mLocalDerivatives[1], mLocalDerivatives[2], mLocalDerivatives[3],
           mGlobalDerivatives[chipId * mNDofPerSensor + 0],
           mGlobalDerivatives[chipId * mNDofPerSensor + 1],
           mGlobalDerivatives[chipId * mNDofPerSensor + 2],
           mGlobalDerivatives[chipId * mNDofPerSensor + 3],
           mAlignPoint->getLocalMeasuredPosition().Y(),
           mAlignPoint->getLocalMeasuredPositionSigma().Y());
    }
    mMillepede->SetLocalEquation(
      mGlobalDerivatives,
      mLocalDerivatives,
      mAlignPoint->getLocalMeasuredPosition().Y(),
      mAlignPoint->getLocalMeasuredPositionSigma().Y());
  } else {
    mCounterLocalEquationFailed++;
  }

  return success;
}

//__________________________________________________________________________
bool TracksToRecords::setLocalEquationZ()
{
  if (!mAlignPoint->isAlignPointSet()) {
    LOGF(error,
         "TracksToRecords::setLocalEquationZ() - no align point coordinates set !");
    return false;
  }
  if (!mAlignPoint->isGlobalDerivativeDone())
    return false;
  if (!mAlignPoint->isLocalDerivativeDone())
    return false;

  bool success = true;

  // clean slate for the local equation for this measurement

  success &= resetGlocalDerivative();
  success &= resetLocalDerivative();

  // local derivatives
  // index [0 .. 3] for {dX0, dTx, dY0, dTz}

  success &= setLocalDerivative(0, mAlignPoint->localDerivativeZ().dX0());
  success &= setLocalDerivative(1, mAlignPoint->localDerivativeZ().dTx());
  success &= setLocalDerivative(2, mAlignPoint->localDerivativeZ().dY0());
  success &= setLocalDerivative(3, mAlignPoint->localDerivativeZ().dTy());

  // global derivatives
  // index [0 .. 3] for {dDeltaX, dDeltaY, dDeltaRz, dDeltaZ}

  Int_t chipId = mAlignPoint->getSensorId();
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 0, mAlignPoint->globalDerivativeZ().dDeltaX());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 1, mAlignPoint->globalDerivativeZ().dDeltaY());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 2, mAlignPoint->globalDerivativeZ().dDeltaRz());
  success &= setGlobalDerivative(chipId * mNDofPerSensor + 3, mAlignPoint->globalDerivativeZ().dDeltaZ());

  if (success) {
    if (mCounterUsedTracks < 5) {
      LOGF(debug,
           "TracksToRecords::setLocalEquationZ(): track %i sr %4d local %.3e %.3e %.3e %.3e, global %.3e %.3e %.3e %.3e Z %.3e sigma %.3e",
           mCounterUsedTracks, chipId,
           mLocalDerivatives[0], mLocalDerivatives[1], mLocalDerivatives[2], mLocalDerivatives[3],
           mGlobalDerivatives[chipId * mNDofPerSensor + 0],
           mGlobalDerivatives[chipId * mNDofPerSensor + 1],
           mGlobalDerivatives[chipId * mNDofPerSensor + 2],
           mGlobalDerivatives[chipId * mNDofPerSensor + 3],
           mAlignPoint->getLocalMeasuredPosition().Z(),
           mAlignPoint->getLocalMeasuredPositionSigma().Z());
    }
    mMillepede->SetLocalEquation(
      mGlobalDerivatives,
      mLocalDerivatives,
      mAlignPoint->getLocalMeasuredPosition().Z(),
      mAlignPoint->getLocalMeasuredPositionSigma().Z());
  } else {
    mCounterLocalEquationFailed++;
  }

  return success;
}
