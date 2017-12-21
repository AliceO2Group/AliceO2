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
#include "Framework/WorkflowSpec.h"
#include "Framework/MetricsService.h"
#include "Framework/RootFileService.h"
#include "Framework/AlgorithmSpec.h"

#include "FairMQLogger.h"

#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairPrimaryGenerator.h"
#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"

#include "DetectorsPassive/Cave.h"
#include "Field/MagneticField.h"

#include "DetectorsPassive/Cave.h"
#include "Generators/GeneratorFromFile.h"
#include "TPCSimulation/Detector.h"

using namespace o2::framework;

using DataHeader = o2::header::DataHeader;


#define BOX_GENERATOR 1

namespace o2 {
namespace workflows {

DataProcessorSpec sim_tpc() {
  return {
    "sim_tpc",
    Inputs{},
    {
      OutputSpec{"TPC", "GEN", OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      [](InitContext &setup) {
        int nEvents = setup.options().get<int>("nEvents");
        auto mcEngine = setup.options().get<std::string>("mcEngine");

        // FIXME: this should probably be part of some generic
        //        FairRunInitSpec
        TString dir = getenv("VMCWORKDIR");
        TString geom_dir = dir + "/Detectors/Geometry/";
        gSystem->Setenv("GEOMPATH",geom_dir.Data());

        TString tut_configdir = dir + "/Detectors/gconfig";
        gSystem->Setenv("CONFIG_DIR",tut_configdir.Data());

        // Requiring a file is something which requires IO, and it's therefore
        // delegated to the framework
        auto &rfm = setup.services().get<RootFileService>();
        // FIXME: We should propably have a service for FairRunSim, rather than
        //        for the root files themselves...
        // Output file name
        auto outFile = rfm.format("AliceO2_%s.tpc.mc_%i_event.root", mcEngine.c_str(), nEvents);

        // Parameter file name
        auto parFile = rfm.format("AliceO2_%s.tpc.mc_%i_event.root", mcEngine.c_str(), nEvents);

        // Create simulation run
        FairRunSim* run = new FairRunSim();

        run->SetName(mcEngine.c_str());
        run->SetOutputFile(outFile.c_str()); // Output file
        FairRuntimeDb* rtdb = run->GetRuntimeDb();

        // Create media
        run->SetMaterials("media.geo"); // Materials

        // Create geometry
        o2::Passive::Cave* cave = new o2::Passive::Cave("CAVE");
        cave->SetGeometryFileName("cave.geo");
        run->AddModule(cave);

        o2::field::MagneticField *magField = new o2::field::MagneticField("Maps","Maps", -1., -1., o2::field::MagFieldParam::k5kG);
        run->SetField(magField);

        // ===| Add TPC |============================================================
        o2::TPC::Detector* tpc = new o2::TPC::Detector(kTRUE);
        tpc->SetGeoFileName("TPCGeometry.root");
        run->AddModule(tpc);

        // Create PrimaryGenerator
        FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
#ifdef BOX_GENERATOR
        FairBoxGenerator* boxGen = new FairBoxGenerator(211, 10); /*protons*/

        //boxGen->SetThetaRange(0.0, 90.0);
        boxGen->SetEtaRange(-0.9,0.9);
        boxGen->SetPRange(0.1, 5);
        boxGen->SetPhiRange(0., 360.);
        boxGen->SetDebug(kTRUE);

        primGen->AddGenerator(boxGen);
#else
        // reading the events from a kinematics file (produced by AliRoot)
        auto extGen = new o2::eventgen::GeneratorFromFile(params.get<std::string>("extKinFile"));
        extGen->SetStartEvent(params.get<int>("startEvent"));
        primGen->AddGenerator(extGen);
#endif

        run->SetGenerator(primGen);

        // store track trajectories
        // run->SetStoreTraj(kTRUE);

        // Initialize simulation run
        run->Init();

        // Runtime database
        Bool_t kParameterMerged = kTRUE;
        FairParRootFileIo* parOut = new FairParRootFileIo(kParameterMerged);
        parOut->open(parFile.c_str());
        rtdb->setOutput(parOut);
        rtdb->saveOutput();
        rtdb->print();
        run->Run(nEvents);

        static bool once = true;

        // This is the actual inner loop for the device
        return [run,nEvents](ProcessingContext &ctx) {
                  if (!once) {
                    run->Run(nEvents);
                    once = true;
                  } else {
                    sleep(1);
                  }
                 // FIXME: After we run we should readback events
                 // and push them as messages, for the next stage of
                 // processing.
              };
        }
      },
    Options{
      {"mcEngine", VariantType::String, "TGeant3", {"Engine to use"}},
      {"nEvents", VariantType::Int, 10, {"Events to process"}},
      {"extKinFile", VariantType::String, "Kinematics.root", {"name of kinematics file for event generator from file (when applicable)"}},
      {"startEvent", VariantType::Int, 2, {"Events to skip"}}
    }
    };
  };
} // namespace workflows
} // namespace o2
