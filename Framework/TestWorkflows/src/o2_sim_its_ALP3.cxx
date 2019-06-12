// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/WorkflowSpec.h"
#include <Monitoring/Monitoring.h>
#include "Framework/AlgorithmSpec.h"
#include "Framework/Logger.h"

#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairPrimaryGenerator.h"
#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"

#include "DetectorsPassive/Cave.h"
#include "Field/MagneticField.h"
#include "ITSSimulation/Detector.h"
#include <cstdlib>
#include <cstdio>

using namespace o2::framework;

using DataHeader = o2::header::DataHeader;

double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

namespace o2 {
namespace workflows {

DataProcessorSpec sim_its_ALP3() {
  return {
    "sim_its_ALP3",
    Inputs{},
    Outputs{
      OutputSpec{"ITS", "HITS"}
    },
    AlgorithmSpec{
      [](InitContext &setup) {
        Int_t nEvents = 10;
        TString mcEngine = "TGeant3";

        TString dir = getenv("VMCWORKDIR");
        TString geom_dir = dir + "/Detectors/Geometry/";
        gSystem->Setenv("GEOMPATH",geom_dir.Data());


        TString tut_configdir = dir + "/Detectors/gconfig";
        gSystem->Setenv("CONFIG_DIR",tut_configdir.Data());

        // Output file name
        char fileout[100];
        sprintf(fileout, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
        TString outFile = fileout;

        // Parameter file name
        char filepar[100];
        sprintf(filepar, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
        TString parFile = filepar;

        FairRunSim *run = new FairRunSim();
        run->SetName(mcEngine);
        run->SetOutputFile(outFile); // Output file
        FairRuntimeDb* rtdb = run->GetRuntimeDb();

        // Create media
        run->SetMaterials("media.geo"); // Materials

        // Create geometry
        o2::passive::Cave* cave = new o2::passive::Cave("CAVE");
        cave->SetGeometryFileName("cave.geo");
        run->AddModule(cave);

        /*FairConstField field;
         field.SetField(0., 0., 5.); //in kG
         field.SetFieldRegion(-5000.,5000.,-5000.,5000.,-5000.,5000.); //in c
        */
        o2::field::MagneticField field("field","field +5kG");
        run->SetField(&field);

        o2::its::Detector* its = new o2::its::Detector(kTRUE);
        run->AddModule(its);

        // Create PrimaryGenerator
        FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
        FairBoxGenerator* boxGen = new FairBoxGenerator(211, 100); //pions

        //boxGen->SetThetaRange(0.0, 90.0);
        boxGen->SetEtaRange(-0.9,0.9);
        boxGen->SetPtRange(1, 1.01);
        boxGen->SetPhiRange(0., 360.);
        boxGen->SetDebug(kFALSE);

        primGen->AddGenerator(boxGen);

        run->SetGenerator(primGen);

        // Initialize simulation run
        run->Init();

        // Runtime database
        Bool_t kParameterMerged = kTRUE;
        FairParRootFileIo* parOut = new FairParRootFileIo(kParameterMerged);
        parOut->open(parFile.Data());
        rtdb->setOutput(parOut);
        rtdb->saveOutput();
        rtdb->print();

        // This is the actual inner loop for the device
        return [run](ProcessingContext &ctx) {
                 run->Run(10);
                 // FIXME: After we run we should readback events
                 // and push them as messages, for the next stage of
                 // processing.
              };
        }
      }
    };
  };
} // namespace workflows
} // namespace o2
