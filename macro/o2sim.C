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

#include "build_geometry.C"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Generators/PrimaryGenerator.h>
#include <Generators/GeneratorFactory.h>
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include <SimConfig/SimConfig.h>
#include <SimConfig/SimParams.h>
#include <CommonUtils/ConfigurableParam.h>
#include <CommonUtils/RngHelper.h>
#include <TStopwatch.h>
#include <memory>
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include <SimSetup/SimSetup.h>
#include <Steer/O2RunSim.h>
#include <DetectorsBase/MaterialManager.h>
#include <CCDB/BasicCCDBManager.h>
#include <CommonUtils/NameConf.h>
#include "DetectorsBase/Aligner.h"
#include <FairRootFileSink.h>
#include <unistd.h>
#include <sstream>
#endif
#include "migrateSimFiles.C"

void check_notransport()
{
  // Sometimes we just want to inspect
  // the generator kinematics and prohibit any transport.

  // We allow users to give the noGeant option. In this case
  // we set the geometry cuts to almost zero. Other adjustments (disable physics procs) could
  // be done on top.
  // This is merely offered for user convenience as it can be done from outside as well.
  auto& confref = o2::conf::SimConfig::Instance();
  if (confref.isNoGeant()) {
    LOG(info) << "Initializing without Geant transport by applying very tight geometry cuts";
    o2::conf::ConfigurableParam::setValue("SimCutParams", "maxRTracking", 0.0000001);    // 1 nanometer of tracking
    o2::conf::ConfigurableParam::setValue("SimCutParams", "maxAbsZTracking", 0.0000001); // 1 nanometer of tracking
    // TODO: disable physics processes for material sitting at the vertex
  }
}

FairRunSim* o2sim_init(bool asservice, bool evalmat = false)
{
  auto& confref = o2::conf::SimConfig::Instance();
  // initialize CCDB service
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  // fix the timestamp early
  uint64_t timestamp = confref.getTimestamp();
  // fix or check timestamp based on given run number if any
  if (confref.getRunNumber() != -1) {
    // if we have a run number we should fix or check the timestamp

    // fetch the actual timestamp ranges for this run
    auto soreor = ccdbmgr.getRunDuration(confref.getRunNumber());
    if (confref.getConfigData().mTimestampMode == o2::conf::kNow) {
      timestamp = soreor.first;
      LOG(info) << "Fixing timestamp to " << timestamp << " based on run number";
      // communicate decision back to sim config object
      confref.getConfigData().mTimestampMode = o2::conf::kRun;
      confref.getConfigData().mTimestamp = timestamp;
    } else if (confref.getConfigData().mTimestampMode == o2::conf::kManual && (timestamp < soreor.first || timestamp > soreor.second)) {
      LOG(error) << "The given timestamp is incompatible with the given run number";
    }
  }
  ccdbmgr.setTimestamp(timestamp);
  ccdbmgr.setURL(confref.getConfigData().mCCDBUrl);
  // try to verify connection
  if (!ccdbmgr.isHostReachable()) {
    LOG(error) << "Could not setup CCDB connection";
  } else {
    LOG(info) << "Initialized CCDB Manager at URL: " << ccdbmgr.getURL();
    LOG(info) << "Initialized CCDB Manager with timestamp : " << ccdbmgr.getTimestamp();
  }

  check_notransport();

  // update the parameters from an INI/JSON file, if given (overrides code-based version)
  o2::conf::ConfigurableParam::updateFromFile(confref.getConfigFile());

  // update the parameters from stuff given at command line (overrides file-based version)
  o2::conf::ConfigurableParam::updateFromString(confref.getKeyValueString());

  // write the final configuration file
  o2::conf::ConfigurableParam::writeINI(o2::base::NameConf::getMCConfigFileName(confref.getOutPrefix()));

  // set seed
  auto seed = o2::utils::RngHelper::setGRandomSeed(confref.getStartSeed());
  LOG(info) << "RNG INITIAL SEED " << seed;

  auto genconfig = confref.getGenerator();
  FairRunSim* run = new o2::steer::O2RunSim(asservice, evalmat);
  run->SetImportTGeoToVMC(false); // do not import TGeo to VMC since the latter is built together with TGeo
  run->SetSimSetup([confref]() { o2::SimSetup::setup(confref.getMCEngine().c_str()); });
  run->SetRunId(timestamp);

  auto pid = getpid();
  std::stringstream s;
  s << confref.getOutPrefix();
  if (asservice) {
    s << "_" << pid;
  }
  s << ".root";

  std::string outputfilename = s.str();
  run->SetSink(new FairRootFileSink(outputfilename.c_str())); // Output file
  run->SetName(confref.getMCEngine().c_str()); // Transport engine
  run->SetIsMT(confref.getIsMT());             // MT mode

  /** set event header **/
  auto header = new o2::dataformats::MCEventHeader();
  run->SetMCEventHeader(header);

  // construct geometry / including magnetic field
  auto flg = TGeoManager::LockDefaultUnits(false);
  TGeoManager::SetDefaultUnits(TGeoManager::kRootUnits);
  TGeoManager::LockDefaultUnits(flg);
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

  o2::detectors::DetID::mask_t detMask{};
  o2::detectors::DetID::mask_t readoutDetMask{};
  {
    auto& modulelist = o2::conf::SimConfig::Instance().getActiveModules();
    for (const auto& md : modulelist) {
      int id = o2::detectors::DetID::nameToID(md.c_str());
      if (id >= o2::detectors::DetID::First) {
        detMask |= o2::detectors::DetID::getMask(id);
        if (isReadout(md)) {
          readoutDetMask |= o2::detectors::DetID::getMask(id);
        }
      }
    }
    if (readoutDetMask.none()) {
      LOG(info) << "Hit creation disabled for all detectors";
    }
    // somewhat ugly, but this is the most straighforward way to make sure the detectors to align
    // don't include detectors which are not activated
    auto& aligner = o2::base::Aligner::Instance();
    auto detMaskAlign = aligner.getDetectorsMask() & detMask;
    aligner.setValue(fmt::format("{}.mDetectors", aligner.getName()), o2::detectors::DetID::getNames(detMaskAlign, ','));
  }

  // run init
  run->Init();

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
  // add ALICE particles to TDatabasePDG singleton
  o2::O2DatabasePDG::addALICEParticles(TDatabasePDG::Instance());

  long runStart = confref.getTimestamp(); // this will signify "time of this MC" (might not coincide with start of Run)
  {
    // store GRPobject
    o2::parameters::GRPObject grp;
    if (confref.getRunNumber() != -1) {
      grp.setRun(confref.getRunNumber());
    } else {
      grp.setRun(run->GetRunId());
    }
    uint64_t runStart = timestamp;
    grp.setTimeStart(runStart);
    grp.setTimeEnd(runStart + 3600000);
    grp.setDetsReadOut(readoutDetMask);
    // CTP is not a physical detector, just flag in the GRP if requested
    if (isReadout("CTP")) {
      grp.addDetReadOut(o2::detectors::DetID::CTP);
    }

    grp.print();
    printf("VMC: %p\n", TVirtualMC::GetMC());
    auto field = dynamic_cast<o2::field::MagneticField*>(run->GetField());
    if (field) {
      o2::units::Current_t currDip = field->getCurrentDipole();
      o2::units::Current_t currL3 = field->getCurrentSolenoid();
      grp.setL3Current(currL3);
      grp.setDipoleCurrent(currDip);
      grp.setFieldUniformity(field->IsUniform());
    }
    // save
    std::string grpfilename = o2::base::NameConf::getGRPFileName(confref.getOutPrefix());
    TFile grpF(grpfilename.c_str(), "recreate");
    grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
  }
  // create GRPECS object
  {
    o2::parameters::GRPECSObject grp;
    grp.setRun(run->GetRunId());
    grp.setTimeStart(runStart);
    grp.setTimeEnd(runStart + 3600000);
    grp.setNHBFPerTF(128); // might be overridden later
    grp.setDetsReadOut(readoutDetMask);
    if (isReadout("CTP")) {
      grp.addDetReadOut(o2::detectors::DetID::CTP);
    }
    grp.setIsMC(true);
    grp.setRunType(o2::parameters::GRPECSObject::RunType::PHYSICS);
    // grp.setDataPeriod("mc"); // decide what to put here
    std::string grpfilename = o2::base::NameConf::getGRPECSFileName(confref.getOutPrefix());
    TFile grpF(grpfilename.c_str(), "recreate");
    grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
  }
  // create GRPMagField object
  {
    o2::parameters::GRPMagField grp;
    auto field = dynamic_cast<o2::field::MagneticField*>(run->GetField());
    if (!field) {
      LOGP(fatal, "Failed to get magnetic field from the FairRunSim");
    }
    o2::units::Current_t currDip = field->getCurrentDipole();
    o2::units::Current_t currL3 = field->getCurrentSolenoid();
    grp.setL3Current(currL3);
    grp.setDipoleCurrent(currDip);
    grp.setFieldUniformity(field->IsUniform());

    std::string grpfilename = o2::base::NameConf::getGRPMagFieldFileName(confref.getOutPrefix());
    TFile grpF(grpfilename.c_str(), "recreate");
    grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
  }
  // create GRPLHCIF object (just a placeholder, bunch filling will be set in digitization)
  {
    o2::parameters::GRPLHCIFData grp;
    // eventually we need to set the beam info from the generator, at the moment put some plausible values
    grp.setFillNumberWithTime(runStart, 0);         // RS FIXME
    grp.setInjectionSchemeWithTime(runStart, "");   // RS FIXME
    grp.setBeamEnergyPerZWithTime(runStart, 6.8e3); // RS FIXME
    grp.setAtomicNumberB1WithTime(runStart, 1.);    // RS FIXME
    grp.setAtomicNumberB2WithTime(runStart, 1.);    // RS FIXME
    grp.setCrossingAngleWithTime(runStart, 0.);     // RS FIXME
    grp.setBeamAZ();

    std::string grpfilename = o2::base::NameConf::getGRPLHCIFFileName(confref.getOutPrefix());
    TFile grpF(grpfilename.c_str(), "recreate");
    grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
  }

  // print summary about cuts and processes used
  auto& matmgr = o2::base::MaterialManager::Instance();
  std::ofstream cutfile(o2::base::NameConf::getCutProcFileName(confref.getOutPrefix()));
  matmgr.printCuts(cutfile);
  matmgr.printProcesses(cutfile);

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  // extract max memory usage for init
  FairSystemInfo sysinfo;
  LOG(info) << "Init: Real time " << rtime << " s, CPU time " << ctime << "s";
  LOG(info) << "Init: Memory used " << sysinfo.GetMaxMemory() << " MB";

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

  LOG(info) << "Macro finished succesfully.";
  LOG(info) << "Real time " << rtime << " s, CPU time " << ctime << "s";
  LOG(info) << "Memory used " << sysinfo.GetMaxMemory() << " MB";

  // migrate to file format where hits sit in separate files
  // (Note: The parallel version is doing this intrinsically;
  //  The serial version uses FairRootManager IO which handles a common file IO for all outputs)
  if (!asservice) {
    LOG(info) << "Migrating simulation output to separate hit file format";
    migrateSimFiles(confref.getOutPrefix().c_str());
  }
}

// asservice: in a parallel device-based context?
void o2sim(bool asservice = false, bool evalmat = false)
{
  auto run = o2sim_init(asservice, evalmat);
  o2sim_run(run, asservice);
  delete run;
}
