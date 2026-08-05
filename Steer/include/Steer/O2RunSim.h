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

// This file is an adaption of FairRoot::FairRunAna commit a6c5cfbe143d3391e (dev branch)
// created 28.9.2017 by S. Wenzel

#ifndef O2_O2RUNSIM_H
#define O2_O2RUNSIM_H

#include "FairRunSim.h" // for FairRunSim
#include "Rtypes.h"     // for Bool_t, Double_t, UInt_t, etc
#include <iostream>

#include "FairField.h"            // for FairField
#include "FairFileHeader.h"       // for FairFileHeader
#include <fairlogger/Logger.h>    // for FairLogger, MESSAGE_ORIGIN
#include "FairMCEventHeader.h"    // for FairMCEventHeader
#include "FairModule.h"           // for FairModule
#include "FairPrimaryGenerator.h" // for FairPrimaryGenerator
#include "FairRootManager.h"      // for FairRootManager
#include "FairRunIdGenerator.h"   // for FairRunIdGenerator
#include "FairTask.h"             // for FairTask
#include <Steer/O2MCApplication.h>

#include <Steer/O2MCApplicationEvalMat.h>

namespace o2
{
namespace steer
{

class O2RunSim : public FairRunSim
{
 public:
  O2RunSim(bool devicemode, bool evalmat) : FairRunSim(), mDeviceMode(devicemode), mEvalMat(evalmat) {}
  ~O2RunSim() override = default;

  void Init() final
  {
    LOG(info) << "O2RUNSIM SPECIFIC INIT CALLED";

    fRootManager->InitSink();

    // No FairGeoLoader here: ALICE never loads an ASCII media file, and the one
    // consumer that forced the singleton to exist - the media loop in
    // FairMCApplication::ConstructOpGeometry() - has been removed.

    if (mDeviceMode) {
      fApp = new O2MCApplication("Fair", "The Fair VMC App", ListOfModules, MatFname);
    } else {
      if (!mEvalMat) {
        fApp = new O2MCApplicationBase("Fair", "The Fair VMC App", ListOfModules, MatFname);
      } else {
        fApp = new O2MCApplicationEvalMat("Fair", "The Fair VMC App", ListOfModules, MatFname);
      }
    }

    fApp->SetGenerator(fGen);

    FairRunIdGenerator genid;
    fRunId = genid.generateId();

    fFileHeader->SetRunId(fRunId);

    // Set global Parameter Info
    if (fPythiaDecayer) {
      fApp->SetPythiaDecayer(fPythiaDecayer);
      if (fPythiaDecayerConfig) {
        fApp->SetPythiaDecayerConfig(fPythiaDecayerConfig);
      }
    }
    if (fUserDecay) {
      fApp->SetUserDecay(fUserDecay);
      if (fUserDecayConfig) {
        fApp->SetUserDecayConfig(fUserDecayConfig);
      }
    }

    if (fField) {
      fField->Init();
    }
    fApp->SetField(fField);

    fSimSetup();
    fApp->InitMC("foo", "bar");
    fRootManager->WriteFileHeader(fFileHeader);
  }

  void Run(int n = 0, int b = 0) final
  {
    FairRunSim::Run(n, b);
  }

 private:
  bool mDeviceMode{false};
  bool mEvalMat{false};

  ClassDefOverride(O2RunSim, 0);
};
} // namespace steer
} // namespace o2

#endif //O2_O2RUNANA_H
