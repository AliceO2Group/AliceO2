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
#include <Generators/Pythia8Generator.h>
#endif
#include <Generators/GeneratorTGenerator.h>
#ifdef GENERATORS_WITH_HEPMC3
#include <Generators/GeneratorHepMC.h>
#endif
#include <Generators/BoxGunParam.h>
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
  auto genconfig = conf.getGenerator();
  if (genconfig.compare("boxgen") == 0) {
    // a simple "box" generator configurable via BoxGunparam
    auto& boxparam = BoxGunParam::Instance();
    LOG(INFO) << "Init box generator with following parameters";
    LOG(INFO) << boxparam;
    auto boxGen = new FairBoxGenerator(boxparam.pdg, boxparam.number);
    boxGen->SetEtaRange(boxparam.eta[0], boxparam.eta[1]);
    boxGen->SetPRange(boxparam.prange[0], boxparam.prange[1]);
    boxGen->SetPhiRange(boxparam.phirange[0], boxparam.phirange[1]);
    boxGen->SetDebug(boxparam.debug);

    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwmugen") == 0) {
    // a simple "box" generator for forward muons
    LOG(INFO) << "Init box forward muons generator";
    auto boxGen = new FairBoxGenerator(13, 1); /* mu- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(100.0, 100.0);
    boxGen->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("hmpidgun") == 0) {
    // a simple "box" generator for forward muons
    LOG(INFO) << "Init hmpid gun generator";
    auto boxGen = new FairBoxGenerator(-211, 100); /* mu- */
    boxGen->SetEtaRange(-0.5, 0.5);
    boxGen->SetPRange(2, 5.0);
    boxGen->SetPhiRange(-5., 60.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwpigen") == 0) {
    // a simple "box" generator for forward pions
    LOG(INFO) << "Init box forward muons generator";
    auto boxGen = new FairBoxGenerator(-211, 1); /* pi- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(7.0, 7.0);
    boxGen->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwrootino") == 0) {
    // a simple "box" generator for forward rootinos
    LOG(INFO) << "Init box forward rootinos generator";
    auto boxGen = new FairBoxGenerator(0, 1); /* mu- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(1, 5);
    boxGen->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("zdcgen") == 0) {
    // a simple "box" generator for forward neutrons
    LOG(INFO) << "Init box forward zdc generator";
    auto boxGenC = new FairBoxGenerator(2212, 500); /* neutrons */
    boxGenC->SetEtaRange(-8.0, -9999);
    boxGenC->SetPRange(10, 500);
    boxGenC->SetPhiRange(0., 360.);
    auto boxGenA = new FairBoxGenerator(2212, 500); /* neutrons */
    boxGenA->SetEtaRange(8.0, 9999);
    boxGenA->SetPRange(10, 500);
    boxGenA->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGenC);
    primGen->AddGenerator(boxGenA);
  } else if (genconfig.compare("fddgen") == 0) {
    LOG(INFO) << "Init box FDD generator";
    auto boxGenFDC = new FairBoxGenerator(13, 1000);
    boxGenFDC->SetEtaRange(-7.0, -4.8);
    boxGenFDC->SetPRange(10, 500);
    boxGenFDC->SetPhiRange(0., 360.);
    auto boxGenFDA = new FairBoxGenerator(13, 1000);
    boxGenFDA->SetEtaRange(4.9, 6.3);
    boxGenFDA->SetPRange(10, 500);
    boxGenFDA->SetPhiRange(0., 360.);
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
  } else if (genconfig.compare("pythia8") == 0) {
    // pythia8 pp
    // configures pythia for min.bias pp collisions at 14 TeV
    // TODO: make this configurable
    auto py8Gen = new o2::eventgen::Pythia8Generator();
    py8Gen->SetParameters("Beams:idA 2212");       // p
    py8Gen->SetParameters("Beams:idB 2212");       // p
    py8Gen->SetParameters("Beams:eCM 14000.");     // [GeV]
    py8Gen->SetParameters("SoftQCD:inelastic on"); // all inelastic processes
    py8Gen->SetParameters("ParticleDecays:tau0Max 0.001");
    py8Gen->SetParameters("ParticleDecays:limitTau0 on");
    primGen->AddGenerator(py8Gen);
  } else if (genconfig.compare("pythia8hf") == 0) {
    // pythia8 pp (HF production)
    // configures pythia for HF production in pp collisions at 14 TeV
    // TODO: make this configurable
    auto py8Gen = new o2::eventgen::Pythia8Generator();
    py8Gen->SetParameters("Beams:idA 2212");   // p
    py8Gen->SetParameters("Beams:idB 2212");   // p
    py8Gen->SetParameters("Beams:eCM 14000."); // [GeV]
    py8Gen->SetParameters("HardQCD:hardccbar on");
    py8Gen->SetParameters("HardQCD:hardbbbar on");
    py8Gen->SetParameters("ParticleDecays:tau0Max 0.001");
    py8Gen->SetParameters("ParticleDecays:limitTau0 on");
    primGen->AddGenerator(py8Gen);
  } else if (genconfig.compare("pythia8hi") == 0) {
    // pythia8 heavy-ion
    // exploits pythia8 heavy-ion machinery (available from v8.230)
    // configures pythia for min.bias Pb-Pb collisions at 5.52 TeV
    // TODO: make this configurable
    auto py8Gen = new o2::eventgen::Pythia8Generator();
    py8Gen->SetParameters("Beams:idA 1000822080");                                      // Pb ion
    py8Gen->SetParameters("Beams:idB 1000822080");                                      // Pb ion
    py8Gen->SetParameters("Beams:eCM 5520.0");                                          // [GeV]
    py8Gen->SetParameters("HeavyIon:SigFitNGen 0");                                     // valid for Pb-Pb 5520 only
    py8Gen->SetParameters("HeavyIon:SigFitDefPar 14.82,1.82,0.25,0.0,0.0,0.0,0.0,0.0"); // valid for Pb-Pb 5520 only
    py8Gen->SetParameters(
      ("HeavyIon:bWidth " + std::to_string(conf.getBMax())).c_str()); // impact parameter from 0-x [fm]
    py8Gen->SetParameters("ParticleDecays:tau0Max 0.001");
    py8Gen->SetParameters("ParticleDecays:limitTau0 on");
    primGen->AddGenerator(py8Gen);
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
}

} // end namespace eventgen
} // end namespace o2
