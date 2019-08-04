// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "FairBaseParSet.h"       // for FairBaseParSet
#include "FairGeoParSet.h"        // for FairGeoParSet
#include "FairField.h"            // for FairField
#include "FairFileHeader.h"       // for FairFileHeader
#include "FairGeoInterface.h"     // for FairGeoInterface
#include "FairGeoLoader.h"        // for FairGeoLoader
#include "FairLogger.h"           // for FairLogger, MESSAGE_ORIGIN
#include "FairMCEventHeader.h"    // for FairMCEventHeader
#include "FairMesh.h"             // for FairMesh
#include "FairModule.h"           // for FairModule
#include "FairParSet.h"           // for FairParSet
#include "FairPrimaryGenerator.h" // for FairPrimaryGenerator
#include "FairRootManager.h"      // for FairRootManager
#include "FairRunIdGenerator.h"   // for FairRunIdGenerator
#include "FairRuntimeDb.h"        // for FairRuntimeDb
#include "FairTask.h"             // for FairTask
#include "FairTrajFilter.h"       // for FairTrajFilter
#include "TRandom.h"
#include <Steer/O2MCApplication.h>

namespace o2
{
namespace steer
{

class O2RunSim : public FairRunSim
{
 public:
  O2RunSim(bool devicemode) : FairRunSim(), mDeviceMode(devicemode) {}
  ~O2RunSim() override = default;

  void Init() final
  {
    LOG(INFO) << "O2RUNSIM SPECIFIC INIT CALLED";

    fRootManager->InitSink();

    // original FairRunSim follows
    FairGeoLoader* loader = new FairGeoLoader(fLoaderName->Data(), "Geo Loader");
    FairGeoInterface* GeoInterFace = loader->getGeoInterface();
    GeoInterFace->SetNoOfSets(ListOfModules->GetEntries());
    GeoInterFace->setMediaFile(MatFname.Data());
    GeoInterFace->readMedia();

    if (mDeviceMode) {
      fApp = new O2MCApplication("Fair", "The Fair VMC App", ListOfModules, MatFname);
    } else {
      fApp = new O2MCApplicationBase("Fair", "The Fair VMC App", ListOfModules, MatFname);
    }

    fApp->SetGenerator(fGen);

    // Add a Generated run ID to the FairRunTimeDb
    FairRunIdGenerator genid;
    // FairRuntimeDb *rtdb= GetRuntimeDb();
    fRunId = genid.generateId();
    fRtdb->addRun(fRunId);

    fFileHeader->SetRunId(fRunId);
    /** This call will create the container if it does not exist*/
    FairBaseParSet* par = dynamic_cast<FairBaseParSet*>(fRtdb->getContainer("FairBaseParSet"));
    if (par) {
      par->SetDetList(GetListOfModules());
      par->SetGen(GetPrimaryGenerator());
      par->SetBeamMom(fBeamMom);
    }

    /** This call will create the container if it does not exist*/
    FairGeoParSet* geopar = dynamic_cast<FairGeoParSet*>(fRtdb->getContainer("FairGeoParSet"));
    if (geopar) {
      geopar->SetGeometry(gGeoManager);
    }

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
    SetFieldContainer();

    TList* containerList = fRtdb->getListOfContainers();
    TIter next(containerList);
    FairParSet* cont;
    TObjArray* ContList = new TObjArray();
    while ((cont = dynamic_cast<FairParSet*>(next()))) {
      ContList->Add(new TObjString(cont->GetName()));
    }
    if (par) {
      par->SetContListStr(ContList);
      par->SetRndSeed(gRandom->GetSeed());
      par->setChanged();
      par->setInputVersion(fRunId, 1);
    }
    if (geopar) {
      geopar->setChanged();
      geopar->setInputVersion(fRunId, 1);
    }

    fSimSetup();
    fApp->InitMC("foo", "bar");
    fRootManager->WriteFileHeader(fFileHeader);
  }

  void Run(int n = 0, int b = 0) final
  {
    FairRunSim::Run(n, b);
  }

 private:
  bool mDeviceMode = false;

  ClassDefOverride(O2RunSim, 0);
};
} // namespace steer
} // namespace o2

#endif //O2_O2RUNANA_H
