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
#ifdef GENERATORS_WITH_PYTHIA8
#include <Generators/GeneratorPythia8.h>
#endif
#include <Generators/GeneratorTGenerator.h>
#ifdef GENERATORS_WITH_HEPMC3
#include <Generators/GeneratorHepMC.h>
#endif
#include <Generators/BoxGunParam.h>
#include <Generators/TriggerParticle.h>
#include <Generators/TriggerParticleParam.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TGlobal.h"
#include "TFunction.h"

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
    gen->readFile(config);
    return gen;
  };
#endif

  /** generators **/

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
    auto boxGen = makeBoxGen(13, 100, -2.5, -4.0, 1000, 1000, 0., 360);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("hmpidgun") == 0) {
    // a simple "box" generator for forward muons
    LOG(INFO) << "Init hmpid gun generator";
    auto boxGen = makeBoxGen(-211, 100, -0.5, -0.5, 2, 5, -5, 60);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwpigen") == 0) {
    // a simple "box" generator for forward pions
    LOG(INFO) << "Init box forward muons generator";
    auto boxGen = makeBoxGen(-211, 100, -2.5, -4.5, 7, 7, 0, 360);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwrootino") == 0) {
    // a simple "box" generator for forward rootinos
    LOG(INFO) << "Init box forward rootinos generator";
    auto boxGen = makeBoxGen(0, 1, -2.5, -4.0, 1, 5, 0, 360);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("zdcgen") == 0) {
    // a simple "box" generator for forward neutrons
    LOG(INFO) << "Init box forward/backward zdc generator";
    auto boxGenC = makeBoxGen(2122 /*neutrons*/, 10, -8, -9999, 500, 1000, 0., 360.);
    auto boxGenA = makeBoxGen(2212 /*neutrons*/, 1, 8.0, 9999, 500, 1000, 0., 360.);
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
#ifdef GENERATORS_WITH_HEPMC3
  } else if (genconfig.compare("hepmc") == 0) {
    // external HepMC file
    auto hepmcGen = new o2::eventgen::GeneratorHepMC();
    hepmcGen->setFileName(conf.getHepMCFileName());
    hepmcGen->setVersion(2);
    primGen->AddGenerator(hepmcGen);
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
  } else if (genconfig.compare("extgen") == 0) {
    // external generator via configuration macro
    auto extgen_filename = conf.getExtGeneratorFileName();
    auto extgen_func = conf.getExtGeneratorFuncName();
    if (extgen_func.empty()) {
      auto size = extgen_filename.size();
      auto firstindex = extgen_filename.find_last_of("/") + 1;
      auto lastindex = extgen_filename.find_last_of(".");
      extgen_func = extgen_filename.substr(firstindex < size ? firstindex : 0,
                                           lastindex < size ? lastindex - firstindex : size - firstindex) +
                    "()";
    }
    if (gROOT->LoadMacro(extgen_filename.c_str()) != 0) {
      LOG(FATAL) << "Cannot find " << extgen_filename;
      return;
    }
    /** retrieve FairGenerator **/
    auto extgen_gfunc = extgen_func.substr(0, extgen_func.find_first_of('('));
    if (!gROOT->GetGlobalFunction(extgen_gfunc.c_str())) {
      LOG(FATAL) << "Global function '"
                 << extgen_gfunc
                 << "' not defined";
    }
    if (strcmp(gROOT->GetGlobalFunction(extgen_gfunc.c_str())->GetReturnTypeName(), "FairGenerator*")) {
      LOG(FATAL) << "Global function '"
                 << extgen_gfunc
                 << "' does not return a 'FairGenerator*' type";
    }
    gROOT->ProcessLine(Form("FairGenerator *__extgen = dynamic_cast<FairGenerator *>(%s);", extgen_func.c_str()));
    auto extgen_ptr = (FairGenerator**)gROOT->GetGlobal("__extgen")->GetAddress();
    primGen->AddGenerator(*extgen_ptr);
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

  o2::eventgen::Trigger* trigger = nullptr;

  auto trgconfig = conf.getTrigger();
  if (trgconfig.empty()) {
    return;
  } else if (trgconfig.compare("particle") == 0) {
    auto& param = TriggerParticleParam::Instance();
    LOG(INFO) << "Init trigger \'particle\' with following parameters";
    LOG(INFO) << param;
    auto trg = new TriggerParticle();
    trg->setPDG(param.pdg);
    trg->setPtRange(param.ptMin, param.ptMax);
    trg->setEtaRange(param.etaMin, param.etaMax);
    trg->setPhiRange(param.phiMin, param.phiMax);
    trg->setYRange(param.yMin, param.yMax);
    trigger = trg;
  } else if (trgconfig.compare("external") == 0) {
    // external trigger via configuration macro
    auto external_trigger_filename = conf.getExtTriggerFileName();
    auto external_trigger_func = conf.getExtTriggerFuncName();
    if (external_trigger_func.empty()) {
      auto size = external_trigger_filename.size();
      auto firstindex = external_trigger_filename.find_last_of("/") + 1;
      auto lastindex = external_trigger_filename.find_last_of(".");
      external_trigger_func = external_trigger_filename.substr(firstindex < size ? firstindex : 0,
                                                               lastindex < size ? lastindex - firstindex : size - firstindex) +
                              "()";
    }
    if (gROOT->LoadMacro(external_trigger_filename.c_str()) != 0) {
      LOG(FATAL) << "Cannot find " << external_trigger_filename;
      return;
    }
    /** retrieve Trigger **/
    auto external_trigger_gfunc = external_trigger_func.substr(0, external_trigger_func.find_first_of('('));
    if (!gROOT->GetGlobalFunction(external_trigger_gfunc.c_str())) {
      LOG(FATAL) << "Global function '"
                 << external_trigger_gfunc
                 << "' not defined";
    }
    if (strcmp(gROOT->GetGlobalFunction(external_trigger_gfunc.c_str())->GetReturnTypeName(), "o2::eventgen::Trigger*")) {
      LOG(FATAL) << "Global function '"
                 << external_trigger_gfunc
                 << "' does not return a 'o2::eventgen::Trigger*' type";
    }
    gROOT->ProcessLine(Form("o2::eventgen::Trigger *__external_trigger__ = dynamic_cast<o2::eventgen::Trigger *>(%s);", external_trigger_func.c_str()));
    auto external_trigger_ptr = (o2::eventgen::Trigger**)gROOT->GetGlobal("__external_trigger__")->GetAddress();
    trigger = *external_trigger_ptr;
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
    generator->addTrigger(trigger);
  }
}

} // end namespace eventgen
} // end namespace o2
