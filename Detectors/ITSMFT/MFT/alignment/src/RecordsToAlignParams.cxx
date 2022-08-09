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

/// @file RecordsToAlignParams.cxx

#include <iostream>
#include <sstream>
#include <string>

#include "Framework/Logger.h"
#include "MFTAlignment/AlignSensorHelper.h"
#include "MFTAlignment/RecordsToAlignParams.h"

using namespace o2::mft;

ClassImp(o2::mft::RecordsToAlignParams);

//__________________________________________________________________________
RecordsToAlignParams::RecordsToAlignParams()
  : Aligner(),
    mWithControl(false),
    mNEntriesAutoSave(10000),
    mRecordReader(new MilleRecordReader()),
    mWithConstraintsRecReader(false),
    mConstraintsRecReader(nullptr),
    mMillepede(new MillePede2())
{
  if (mWithConstraintsRecReader) {
    mConstraintsRecReader = new MilleRecordReader();
  }
  LOGF(debug, "RecordsToAlignParams instantiated");
}

//__________________________________________________________________________
RecordsToAlignParams::~RecordsToAlignParams()
{
  if (mConstraintsRecReader) {
    delete mConstraintsRecReader;
  }
  if (mMillepede) {
    delete mMillepede;
  }
  if (mRecordReader) {
    delete mRecordReader;
  }
  LOGF(debug, "RecordsToAlignParams destroyed");
}

//__________________________________________________________________________
void RecordsToAlignParams::init()
{
  if (mIsInitDone) {
    return;
  }

  mMillepede->SetRecordReader(mRecordReader);

  if (mWithConstraintsRecReader) {
    mMillepede->SetConstraintsRecReader(mConstraintsRecReader);
  }

  mMillepede->InitMille(mNumberOfGlobalParam,
                        mNumberOfTrackParam,
                        mChi2CutNStdDev,
                        mResCut,
                        mResCutInitial);

  LOG(info) << "-------------- RecordsToAlignParams configured with -----------------";
  LOGF(info, "Chi2CutNStdDev = %d", mChi2CutNStdDev);
  LOGF(info, "ResidualCutInitial = %.3f", mResCutInitial);
  LOGF(info, "ResidualCut = %.3f", mResCut);
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

  LOGF(info, "RecordsToAlignParams init done");
  mIsInitDone = true;
}

//__________________________________________________________________________
void RecordsToAlignParams::globalFit()
{
  if (!mIsInitDone) {
    LOGF(fatal, "RecordsToAlignParams::globalFit() aborted because init was not done !");
    return;
  }
  if (!mRecordReader || !mRecordReader->isReaderOk() || !mRecordReader->getNEntries()) {
    LOGF(fatal, "RecordsToAlignParams::globalFit() aborted because no data record can be read !");
    return;
  }

  // initialize the file and tree to store chi2 from Millepede LocalFit()

  if (mWithControl) {
    mMillepede->InitChi2Storage(mNEntriesAutoSave);
  }

  // allocate memory in arrays to temporarily store the results of the global fit

  double* params = (double*)malloc(sizeof(double) * mNumberOfGlobalParam);
  double* paramsErrors = (double*)malloc(sizeof(double) * mNumberOfGlobalParam);
  double* paramsPulls = (double*)malloc(sizeof(double) * mNumberOfGlobalParam);

  // initialise the content of each array

  for (int ii = 0; ii < mNumberOfGlobalParam; ii++) {
    params[ii] = 0.;
    paramsErrors[ii] = 0.;
    paramsPulls[ii] = 0.;
  }

  // perform the simultaneous fit of track and alignement parameters

  mMillepede->GlobalFit(params, paramsErrors, paramsPulls);

  if (mWithControl) {
    mMillepede->EndChi2Storage();
  }

  // post-treatment:
  // debug output + save Millepede global fit result in AlignParam vector

  LOGF(info, "RecordsToAlignParams::globalFit() - done, results below");
  LOGF(info, "sensor info, dx (cm), dy (cm), dz (cm), dRz (rad)");

  AlignSensorHelper chipHelper;
  double dRx = 0., dRy = 0., dRz = 0.; // delta rotations
  double dx = 0., dy = 0., dz = 0.;    // delta translations
  bool global = true;                  // delta in global ref. system
  bool withSymName = false;

  for (int chipId = 0; chipId < mNumberOfSensors; chipId++) {
    chipHelper.setSensorOnlyInfo(chipId);
    std::stringstream name = chipHelper.getSensorFullName(withSymName);
    dx = params[chipId * mNDofPerSensor + 0];
    dy = params[chipId * mNDofPerSensor + 1];
    dz = params[chipId * mNDofPerSensor + 3];
    dRz = params[chipId * mNDofPerSensor + 2];
    LOGF(info,
         "%s, %.3e, %.3e, %.3e, %.3e",
         name.str().c_str(), dx, dy, dz, dRz);
    mAlignParams.emplace_back(
      chipHelper.geoSymbolicName(),
      chipHelper.sensorUid(),
      dx, dy, dz,
      dRx, dRy, dRz,
      global);
  }

  // free allocated memory

  free(params);
  free(paramsErrors);
  free(paramsPulls);
}

//__________________________________________________________________________
void RecordsToAlignParams::connectRecordReaderToChain(TChain* ch)
{
  if (mRecordReader) {
    mRecordReader->connectToChain(ch);
  }
}

//__________________________________________________________________________
void RecordsToAlignParams::connectConstraintsRecReaderToChain(TChain* ch)
{
  if (mConstraintsRecReader) {
    mConstraintsRecReader->changeDataBranchName();
    mConstraintsRecReader->connectToChain(ch);
  }
}
