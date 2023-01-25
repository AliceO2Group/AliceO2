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
#include <TTree.h>
#include <TROOT.h>
#include <fairlogger/Logger.h>
#include <FOCALBase/TestbeamAnalysis.h>

using namespace o2::focal;

void TestbeamAnalysis::run()
{

  mCurrentFile = std::unique_ptr<TFile>(TFile::Open(mInputFilename.data(), "READ"));
  if (!mCurrentFile || mCurrentFile->IsZombie()) {
    LOG(error) << "Failed reading input file " << mInputFilename;
  }

  mEventReader = std::make_unique<EventReader>(mCurrentFile->Get<TTree>("o2sim"));
  mCurrentEventNumber = 0;

  gROOT->cd();
  init();

  while (mEventReader->hasNext()) {
    auto currentevent = mEventReader->readNextEvent();
    if (mVerbose) {
      LOG(info) << "Processing event " << mCurrentEventNumber;
    }
    if (currentevent.isInitialized()) {
      process(currentevent);
      mCurrentEventNumber++;
    }
  }

  terminate();
}
