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
    std::cout << "Init box generator\n";
    auto boxGen = new FairBoxGenerator(211, 10); /*protons*/
    boxGen->SetEtaRange(-0.9, 0.9);
    boxGen->SetPRange(0.1, 5);
    boxGen->SetPhiRange(0., 360.);
    boxGen->SetDebug(kTRUE);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwmugen") == 0) {
    // a simple "box" generator for forward muons
    std::cout << "Init box forward muons generator\n";
    auto boxGen = new FairBoxGenerator(13, 1); /* mu- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(100.0, 100.0);
    boxGen->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwpigen") == 0) {
    // a simple "box" generator for forward pions
    std::cout << "Init box forward muons generator\n";
    auto boxGen = new FairBoxGenerator(-211, 1); /* pi- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(7.0, 7.0);
    boxGen->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("fwrootino") == 0) {
    // a simple "box" generator for forward rootinos
    std::cout << "Init box forward rootinos generator\n";
    auto boxGen = new FairBoxGenerator(0, 1); /* mu- */
    boxGen->SetEtaRange(-2.5, -4.0);
    boxGen->SetPRange(1, 5);
    boxGen->SetPhiRange(0., 360.);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("extkin") == 0) {
    // external kinematics
    // needs precense of a kinematics file "Kinematics.root"
    // TODO: make this configurable and check for presence
    auto extGen = new o2::eventgen::GeneratorFromFile(conf.getExtKinematicsFileName().c_str());
    extGen->SetStartEvent(conf.getStartEvent());
    primGen->AddGenerator(extGen);
    std::cout << "using external kinematics\n";
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
  } else {
    LOG(FATAL) << "Invalid generator" << FairLogger::endl;
  }
}

} // end namespace eventgen
} // end namespace o2
