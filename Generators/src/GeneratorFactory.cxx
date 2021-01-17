// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author S. Wenzel - Mai 2018

#include <Generators/GeneratorFactory.h>
#include "FairPrimaryGenerator.h"
#include "FairGenerator.h"
#include "FairBoxGenerator.h"
#include <FairLogger.h>
#include <SimConfig/SimConfig.h>
#include <Generators/GeneratorFromFile.h>
#ifdef GENERATORS_WITH_PYTHIA6
#include <Generators/GeneratorPythia6.h>
#include <Generators/GeneratorPythia6Param.h>
#endif
#ifdef GENERATORS_WITH_PYTHIA8
#include <Generators/GeneratorPythia8.h>
#include <Generators/GeneratorPythia8Param.h>
#endif
#include <Generators/GeneratorTGenerator.h>
#include <Generators/GeneratorExternalParam.h>
#ifdef GENERATORS_WITH_HEPMC3
#include <Generators/GeneratorHepMC.h>
#include <Generators/GeneratorHepMCParam.h>
#endif
#include <Generators/BoxGunParam.h>
#include <Generators/PDG.h>
#include <Generators/TriggerParticle.h>
#include <Generators/TriggerExternalParam.h>
#include <Generators/TriggerParticleParam.h>
#include "CommonUtils/ConfigurationMacroHelper.h"

#include "TRandom.h"

namespace o2
{
namespace eventgen
{

// reusable helper class
// main purpose is to init a FairPrimGen given some (Sim)Config
void GeneratorFactory::setPrimaryGenerator(o2::conf::SimConfig const& conf, FairPrimaryGenerator* primGen)
{
  if (!primGen) {
    LOG(WARNING) << "No primary generator instance; Cannot setup";
    return;
  }

  auto makeBoxGen = [](int pdgid, int mult, double etamin, double etamax, double pmin, double pmax, double phimin, double phimax, bool debug = false) {
    auto gen = new FairBoxGenerator(pdgid, mult);
    gen->SetEtaRange(etamin, etamax);
    gen->SetPRange(pmin, pmax);
    gen->SetPhiRange(phimin, phimax);
    gen->SetDebug(debug);
    return gen;
  };

#ifdef GENERATORS_WITH_PYTHIA8
  auto makePythia8Gen = [](std::string& config) {
    auto gen = new o2::eventgen::GeneratorPythia8();
    LOG(INFO) << "Reading \'Pythia8\' base configuration: " << config << std::endl;
    gen->readFile(config);
    auto seed = (gRandom->GetSeed() % 900000000);
    LOG(INFO) << "Using random seed from gRandom % 900000000: " << seed;
    gen->readString("Random:setSeed on");
    gen->readString("Random:seed " + std::to_string(seed));
    return gen;
  };
#endif

  /** generators **/

  o2::PDG::addParticlesToPdgDataBase();
  auto genconfig = conf.getGenerator();
  if (genconfig.compare("boxgen") == 0) {
    // a simple "box" generator configurable via BoxGunparam
    auto& boxparam = BoxGunParam::Instance();
    LOG(INFO) << "Init generic box generator with following parameters";
    LOG(INFO) << boxparam;
    auto boxGen = makeBoxGen(boxparam.pdg, boxparam.number, boxparam.eta[0], boxparam.eta[1], boxparam.prange[0], boxparam.prange[1], boxparam.phirange[0], boxparam.phirange[1], boxparam.debug);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwmugen") == 0) {
    // a simple "box" generator for forward muons
    LOG(INFO) << "Init box forward muons generator";
    auto boxGen = makeBoxGen(13, 1, -4, -2.5, 50., 50., 0., 360);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("hmpidgun") == 0) {
    // a simple "box" generator for forward muons
    LOG(INFO) << "Init hmpid gun generator";
    auto boxGen = makeBoxGen(-211, 100, -0.5, -0.5, 2, 5, -5, 60);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwpigen") == 0) {
    // a simple "box" generator for forward pions
    LOG(INFO) << "Init box forward pions generator";
    auto boxGen = makeBoxGen(-211, 10, -4, -2.5, 7, 7, 0, 360);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwrootino") == 0) {
    // a simple "box" generator for forward rootinos
    LOG(INFO) << "Init box forward rootinos generator";
    auto boxGen = makeBoxGen(0, 1, -4, -2.5, 1, 5, 0, 360);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("zdcgen") == 0) {
    // a simple "box" generator for forward neutrons
    LOG(INFO) << "Init box forward/backward zdc generator";
    auto boxGenC = makeBoxGen(2112 /*neutrons*/, 1, -8, -9999, 500, 1000, 0., 360.);
    auto boxGenA = makeBoxGen(2112 /*neutrons*/, 1, 8, 9999, 500, 1000, 0., 360.);
    primGen->AddGenerator(boxGenC);
    primGen->AddGenerator(boxGenA);
  } else if (genconfig.compare("emcgenele") == 0) {
    // box generator with one electron per event
    LOG(INFO) << "Init box generator for electrons in EMCAL";
    // using phi range of emcal
    auto elecgen = makeBoxGen(11, 1, -0.67, 0.67, 15, 15, 80, 187);
    primGen->AddGenerator(elecgen);
  } else if (genconfig.compare("emcgenphoton") == 0) {
    LOG(INFO) << "Init box generator for photons in EMCAL";
    auto photongen = makeBoxGen(22, 1, -0.67, 0.67, 15, 15, 80, 187);
    primGen->AddGenerator(photongen);
  } else if (genconfig.compare("fddgen") == 0) {
    LOG(INFO) << "Init box FDD generator";
    auto boxGenFDC = makeBoxGen(13, 1000, -7, -4.8, 10, 500, 0, 360.);
    auto boxGenFDA = makeBoxGen(13, 1000, 4.9, 6.3, 10, 500, 0., 360);
    primGen->AddGenerator(boxGenFDA);
    primGen->AddGenerator(boxGenFDC);
  } else if (genconfig.compare("extkin") == 0) {
    // external kinematics
    // needs precense of a kinematics file "Kinematics.root"
    // TODO: make this configurable and check for presence
    auto extGen = new o2::eventgen::GeneratorFromFile(conf.getExtKinematicsFileName().c_str());
    extGen->SetStartEvent(conf.getStartEvent());
    primGen->AddGenerator(extGen);
    LOG(INFO) << "using external kinematics";
  } else if (genconfig.compare("extkinO2") == 0) {
    // external kinematics from previous O2 output
    auto extGen = new o2::eventgen::GeneratorFromO2Kine(conf.getExtKinematicsFileName().c_str());
    extGen->SetStartEvent(conf.getStartEvent());
    primGen->AddGenerator(extGen);
    LOG(INFO) << "using external O2 kinematics";
#ifdef GENERATORS_WITH_HEPMC3
  } else if (genconfig.compare("hepmc") == 0) {
    // external HepMC file
    auto& param = GeneratorHepMCParam::Instance();
    LOG(INFO) << "Init \'GeneratorHepMC\' with following parameters";
    LOG(INFO) << param;
    auto hepmcGen = new o2::eventgen::GeneratorHepMC();
    hepmcGen->setFileName(param.fileName);
    hepmcGen->setVersion(param.version);
    primGen->AddGenerator(hepmcGen);
#endif
#ifdef GENERATORS_WITH_PYTHIA6
  } else if (genconfig.compare("pythia6") == 0) {
    // pythia6 pp
    // configures pythia6 according to param
    auto& param = GeneratorPythia6Param::Instance();
    LOG(INFO) << "Init \'Pythia6\' generator with following parameters";
    LOG(INFO) << param;
    auto py6Gen = new o2::eventgen::GeneratorPythia6();
    py6Gen->setConfig(param.config);
    py6Gen->setFrame(param.frame);
    py6Gen->setBeam(param.beam);
    py6Gen->setTarget(param.target);
    py6Gen->setWin(param.win);
    primGen->AddGenerator(py6Gen);
#endif
#ifdef GENERATORS_WITH_PYTHIA8
  } else if (genconfig.compare("alldets") == 0) {
    // a simple generator for test purposes - making sure to generate hits
    // in all detectors
    // I compose it of:
    // 1) pythia8
    auto py8config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_inel.cfg";
    auto py8 = makePythia8Gen(py8config);
    primGen->AddGenerator(py8);
    // 2) forward muons
    auto muon = makeBoxGen(13, 100, -2.5, -4.0, 100, 100, 0., 360);
    primGen->AddGenerator(muon);
  } else if (genconfig.compare("pythia8") == 0) {
    auto py8config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_inel.cfg";
    auto py8 = makePythia8Gen(py8config);
    primGen->AddGenerator(py8);
  } else if (genconfig.compare("pythia8hf") == 0) {
    // pythia8 pp (HF production)
    // configures pythia for HF production in pp collisions at 14 TeV
    auto py8config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hf.cfg";
    auto py8 = makePythia8Gen(py8config);
    primGen->AddGenerator(py8);
  } else if (genconfig.compare("pythia8hi") == 0) {
    // pythia8 heavy-ion
    // exploits pythia8 heavy-ion machinery (available from v8.230)
    // configures pythia for min.bias Pb-Pb collisions at 5.52 TeV
    auto py8config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hi.cfg";
    auto py8 = makePythia8Gen(py8config);
    primGen->AddGenerator(py8);
#endif
  } else if (genconfig.compare("external") == 0 || genconfig.compare("extgen") == 0) {
    // external generator via configuration macro
    auto& params = GeneratorExternalParam::Instance();
    LOG(INFO) << "Setting up external generator with following parameters";
    LOG(INFO) << params;
    auto extgen_filename = params.fileName;
    auto extgen_func = params.funcName;
    auto extgen = o2::conf::GetFromMacro<FairGenerator*>(extgen_filename, extgen_func, "FairGenerator*", "extgen");
    if (!extgen) {
      LOG(FATAL) << "Failed to retrieve \'extgen\': problem with configuration ";
    }
    primGen->AddGenerator(extgen);
  } else if (genconfig.compare("toftest") == 0) { // 1 muon per sector and per module
    LOG(INFO) << "Init tof test generator -> 1 muon per sector and per module";
    for (int i = 0; i < 18; i++) {
      for (int j = 0; j < 5; j++) {
        auto boxGen = new FairBoxGenerator(13, 1); /*protons*/
        boxGen->SetEtaRange(-0.8 + 0.32 * j + 0.15, -0.8 + 0.32 * j + 0.17);
        boxGen->SetPRange(9, 10);
        boxGen->SetPhiRange(10 + 20. * i - 1, 10 + 20. * i + 1);
        boxGen->SetDebug(kTRUE);
        primGen->AddGenerator(boxGen);
      }
    }
  } else {
    LOG(FATAL) << "Invalid generator";
  }

  /** triggers **/

  Trigger trigger = nullptr;
  DeepTrigger deeptrigger = nullptr;

  auto trgconfig = conf.getTrigger();
  if (trgconfig.empty()) {
    return;
  } else if (trgconfig.compare("particle") == 0) {
    trigger = TriggerParticle(TriggerParticleParam::Instance());
  } else if (trgconfig.compare("external") == 0) {
    // external trigger via configuration macro
    auto& params = TriggerExternalParam::Instance();
    LOG(INFO) << "Setting up external trigger with following parameters";
    LOG(INFO) << params;
    auto external_trigger_filename = params.fileName;
    auto external_trigger_func = params.funcName;
    trigger = o2::conf::GetFromMacro<o2::eventgen::Trigger>(external_trigger_filename, external_trigger_func, "o2::eventgen::Trigger", "trigger");
    if (!trigger) {
      LOG(INFO) << "Trying to retrieve a \'o2::eventgen::DeepTrigger\' type" << std::endl;
      deeptrigger = o2::conf::GetFromMacro<o2::eventgen::DeepTrigger>(external_trigger_filename, external_trigger_func, "o2::eventgen::DeepTrigger", "deeptrigger");
    }
    if (!trigger && !deeptrigger) {
      LOG(FATAL) << "Failed to retrieve \'external trigger\': problem with configuration ";
    }
  } else {
    LOG(FATAL) << "Invalid trigger";
  }

  /** add trigger to generators **/
  auto generators = primGen->GetListOfGenerators();
  for (int igen = 0; igen < generators->GetEntries(); ++igen) {
    auto generator = dynamic_cast<o2::eventgen::Generator*>(generators->At(igen));
    if (!generator) {
      LOG(FATAL) << "request to add a trigger to an unsupported generator";
      return;
    }
    generator->setTriggerMode(o2::eventgen::Generator::kTriggerOR);
    if (trigger) {
      generator->addTrigger(trigger);
    }
    if (deeptrigger) {
      generator->addDeepTrigger(deeptrigger);
    }
  }
}

} // end namespace eventgen
} // end namespace o2
