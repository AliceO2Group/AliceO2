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
#include <Generators/Pythia8Generator.h>
#include <Generators/GeneratorTGenerator.h>
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
    // a simple "box" generator
    LOG(INFO) << "Init box generator";
    auto boxGen = new FairBoxGenerator(211, 10); /*protons*/
    boxGen->SetEtaRange(-0.9, 0.9);
    boxGen->SetPRange(0.1, 5);
    boxGen->SetPhiRange(0., 360.);
    boxGen->SetDebug(kTRUE);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwmugen") == 0) {
    // a simple "box" generator for forward muons
    LOG(INFO) << "Init box forward muons generator";
    auto boxGen = new FairBoxGenerator(13, 1); /* mu- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(100.0, 100.0);
    boxGen->SetPhiRange(0., 360.);
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
    auto boxGen = new FairBoxGenerator(2212, 1000); /* neutrons */
    boxGen->SetEtaRange(-8.0, -9999);
    boxGen->SetPRange(10, 500);
    boxGen->SetPhiRange(0., 360.);
    boxGen->SetDebug(kTRUE);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("extkin") == 0) {
    // external kinematics
    // needs precense of a kinematics file "Kinematics.root"
    // TODO: make this configurable and check for presence
    auto extGen = new o2::eventgen::GeneratorFromFile(conf.getExtKinematicsFileName().c_str());
    extGen->SetStartEvent(conf.getStartEvent());
    primGen->AddGenerator(extGen);
    LOG(INFO) << "using external kinematics";
  } else if (genconfig.compare("pythia8") == 0) {
    // pythia8 pp
    // configures pythia for min.bias pp collisions at 14 TeV
    // TODO: make this configurable
    auto py8Gen = new o2::eventgen::Pythia8Generator();
    py8Gen->SetParameters("Beams:idA 2212");       // p
    py8Gen->SetParameters("Beams:idB 2212");       // p
    py8Gen->SetParameters("Beams:eCM 14000.");     // [GeV]
    py8Gen->SetParameters("SoftQCD:inelastic on"); // all inelastic processes
    py8Gen->SetParameters("ParticleDecays:xyMax 0.1");
    py8Gen->SetParameters("ParticleDecays:zMax 1.");
    py8Gen->SetParameters("ParticleDecays:limitCylinder on");
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
    py8Gen->SetParameters("ParticleDecays:xyMax 0.1");
    py8Gen->SetParameters("ParticleDecays:zMax 1.");
    py8Gen->SetParameters("ParticleDecays:limitCylinder on");
    primGen->AddGenerator(py8Gen);
  } else if (genconfig.compare("extgen") == 0) {
    // external generator via TGenerator interface
    auto tgen = new o2::eventgen::GeneratorTGenerator();
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
      LOG(FATAL) << "Cannot find " << extgen_filename << FairLogger::endl;
      return;
    }
    /** setup convertion units **/
    if (gROOT->GetGlobal("momentumUnit")) {
      auto ptr = (double*)gROOT->GetGlobal("momentumUnit")->GetAddress();
      tgen->setMomentumUnit(*ptr);
    } else {
      LOG(FATAL) << "Mandatory global variable \'momentumUnit\' not defined";
    }
    if (gROOT->GetGlobal("energyUnit")) {
      auto ptr = (double*)gROOT->GetGlobal("energyUnit")->GetAddress();
      tgen->setEnergyUnit(*ptr);
    } else {
      LOG(FATAL) << "Mandatory global variable \'energyUnit\' not defined";
    }
    if (gROOT->GetGlobal("positionUnit")) {
      auto ptr = (double*)gROOT->GetGlobal("positionUnit")->GetAddress();
      tgen->setPositionUnit(*ptr);
    } else {
      LOG(FATAL) << "Mandatory global variable \'positionUnit\' not defined";
    }
    if (gROOT->GetGlobal("timeUnit")) {
      auto ptr = (double*)gROOT->GetGlobal("timeUnit")->GetAddress();
      tgen->setMomentumUnit(*ptr);
    } else {
      LOG(FATAL) << "Mandatory global variable \'timeUnit\' not defined";
    }
    /** retrieve TGenerator **/
    auto extgen_gfunc = extgen_func.substr(0, extgen_func.find_first_of('('));
    if (!gROOT->GetGlobalFunction(extgen_gfunc.c_str())) {
      LOG(FATAL) << "Global function \'"
                 << extgen_gfunc
                 << "\' not defined";
    }
    if (strcmp(gROOT->GetGlobalFunction(extgen_gfunc.c_str())->GetReturnTypeName(), "TGenerator*")) {
      LOG(FATAL) << "Global function \'"
                 << extgen_gfunc
                 << "\' does not return a \'TGenerator*\' type";
    }
    gROOT->ProcessLine(Form("TGenerator *__extgen = dynamic_cast<TGenerator *>(%s);", extgen_func.c_str()));
    auto extgen_ptr = (TGenerator**)gROOT->GetGlobal("__extgen")->GetAddress();
    tgen->setTGenerator(*extgen_ptr);
    primGen->AddGenerator(tgen);
  } else {
    LOG(FATAL) << "Invalid generator";
  }
}

} // end namespace eventgen
} // end namespace o2
