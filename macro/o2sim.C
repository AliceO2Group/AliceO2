// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "build_geometry.C"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Generators/PrimaryGenerator.h>
#include <Generators/GeneratorFactory.h>
#include <Generators/PDG.h>
#include "SimulationDataFormat/MCEventHeader.h"
#include <SimConfig/SimConfig.h>
#include <SimConfig/ConfigurableParam.h>
#include <CommonUtils/RngHelper.h>
#include <TStopwatch.h>
#include <memory>
#include "DataFormatsParameters/GRPObject.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include <SimSetup/SimSetup.h>
#include <Steer/O2RunSim.h>
#include <DetectorsBase/MaterialManager.h>
#include <unistd.h>
#include <sstream>
#endif

FairRunSim* o2sim_init(bool asservice)
{
  auto& confref = o2::conf::SimConfig::Instance();
  // we can read from CCDB (for the moment faking with a TFile)
  // o2::conf::ConfigurableParam::fromCCDB("params_ccdb.root", runid);

  // update the parameters from an INI/JSON file, if given (overrides code-based version)
  o2::conf::ConfigurableParam::updateFromFile(confref.getConfigFile());

  // update the parameters from stuff given at command line (overrides file-based version)
  o2::conf::ConfigurableParam::updateFromString(confref.getKeyValueString());

  // write the configuration file
  o2::conf::ConfigurableParam::writeINI("o2sim_configuration.ini");

  // we can update the binary CCDB entry something like this ( + timestamp key )
  // o2::conf::ConfigurableParam::toCCDB("params_ccdb.root");

  // set seed
  auto seed = o2::utils::RngHelper::setGRandomSeed(confref.getStartSeed());
  LOG(INFO) << "RNG INITIAL SEED " << seed;

  auto genconfig = confref.getGenerator();
  FairRunSim* run = new o2::steer::O2RunSim(asservice);
  run->SetImportTGeoToVMC(false); // do not import TGeo to VMC since the latter is built together with TGeo
  run->SetSimSetup([confref]() { o2::SimSetup::setup(confref.getMCEngine().c_str()); });

  auto pid = getpid();
  std::stringstream s;
  s << confref.getOutPrefix();
  if (asservice) {
    s << "_" << pid;
  }
  s << ".root";

  std::string outputfilename = s.str();
  run->SetOutputFile(outputfilename.c_str());  // Output file
  run->SetName(confref.getMCEngine().c_str()); // Transport engine
  run->SetIsMT(confref.getIsMT());             // MT mode

  /** set event header **/
  auto header = new o2::dataformats::MCEventHeader();
  run->SetMCEventHeader(header);

  // construct geometry / including magnetic field
  build_geometry(run);

  // setup generator
  auto embedinto_filename = confref.getEmbedIntoFileName();
  auto primGen = new o2::eventgen::PrimaryGenerator();
  if (!embedinto_filename.empty()) {
    primGen->embedInto(embedinto_filename);
  }
  if (!asservice) {
    o2::eventgen::GeneratorFactory::setPrimaryGenerator(confref, primGen);
  }
  run->SetGenerator(primGen);

  // Timer
  TStopwatch timer;
  timer.Start();

  // run init
  run->Init();
  finalize_geometry(run);
  std::stringstream geomss;
  geomss << "O2geometry";
  if (asservice) {
    geomss << "_" << pid;
  }
  geomss << ".root";
  gGeoManager->Export(geomss.str().c_str());
  if (asservice) {
    // create a link of expected file name to actually produced file
    // (deletes link if previously existing)
    unlink("O2geometry.root");
    if (symlink(geomss.str().c_str(), "O2geometry.root") != 0) {
      LOG(ERROR) << "Failed to create simlink to geometry file";
    }
  }
  std::time_t runStart = std::time(nullptr);

  // runtime database
  bool kParameterMerged = true;
  auto rtdb = run->GetRuntimeDb();
  auto parOut = new FairParRootFileIo(kParameterMerged);

  std::stringstream s2;
  s2 << confref.getOutPrefix();
  if (asservice) {
    s2 << "_" << pid;
  }
  s2 << "_par.root";
  std::string parfilename = s2.str();
  parOut->open(parfilename.c_str());
  rtdb->setOutput(parOut);
  rtdb->saveOutput();
  rtdb->print();
  o2::PDG::addParticlesToPdgDataBase(0);

  {
    // store GRPobject
    o2::parameters::GRPObject grp;
    grp.setRun(run->GetRunId());
    grp.setTimeStart(runStart);
    grp.setTimeEnd(std::time(nullptr));
    TObjArray* modArr = run->GetListOfModules();
    TIter next(modArr);
    FairModule* module = nullptr;
    while ((module = (FairModule*)next())) {
      o2::base::Detector* det = dynamic_cast<o2::base::Detector*>(module);
      if (!det) {
        continue; // not a detector
      }
      if (det->GetDetId() < o2::detectors::DetID::First) {
        continue; // passive
      }
      if (det->GetDetId() > o2::detectors::DetID::Last) {
        continue; // passive
      }
      grp.addDetReadOut(o2::detectors::DetID(det->GetDetId()));
    }
    grp.print();
    printf("VMC: %p\n", TVirtualMC::GetMC());
    auto field = dynamic_cast<o2::field::MagneticField*>(run->GetField());
    if (field) {
      o2::units::Current_t currDip = field->getCurrentDipole();
      o2::units::Current_t currL3 = field->getCurrentSolenoid();
      grp.setL3Current(currL3);
      grp.setDipoleCurrent(currDip);
    }
    // save
    std::string grpfilename = confref.getOutPrefix() + "_grp.root";
    TFile grpF(grpfilename.c_str(), "recreate");
    grpF.WriteObjectAny(&grp, grp.Class(), "GRP");
  }
  // todo: save beam information in the grp

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  // extract max memory usage for init
  FairSystemInfo sysinfo;
  LOG(INFO) << "Init: Real time " << rtime << " s, CPU time " << ctime << "s";
  LOG(INFO) << "Init: Memory used " << sysinfo.GetMaxMemory() << " MB";

  return run;
}

// only called from the normal o2sim
void o2sim_run(FairRunSim* run, bool asservice)
{
  TStopwatch timer;
  timer.Start();

  auto& confref = o2::conf::SimConfig::Instance();
  if (!asservice) {
    run->Run(confref.getNEvents());
  } else {
    run->Run(1);
  }

  // Finish
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  // extract max memory usage
  FairSystemInfo sysinfo;

  LOG(INFO) << "Macro finished succesfully.";
  LOG(INFO) << "Real time " << rtime << " s, CPU time " << ctime << "s";
  LOG(INFO) << "Memory used " << sysinfo.GetMaxMemory() << " MB";
}

// asservice: in a parallel device-based context?
void o2sim(bool asservice = false)
{
  auto run = o2sim_init(asservice);
  o2sim_run(run, asservice);
  o2::base::MaterialManager::printContainingMedia("ITSV");
  delete run;
}
