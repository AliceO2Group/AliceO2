// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - January 2020

#include "Generators/GeneratorPythia8.h"
#include <Generators/GeneratorPythia8Param.h>
#include "FairLogger.h"
#include "TClonesArray.h"
#include "TParticle.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

GeneratorPythia8::GeneratorPythia8() : Generator("ALICEo2", "ALICEo2 Pythia8 Generator")
{
  /** default constructor **/
}

/*****************************************************************/

GeneratorPythia8::GeneratorPythia8(const Char_t* name, const Char_t* title) : Generator(name, title)
{
  /** constructor **/
}

/*****************************************************************/

Bool_t GeneratorPythia8::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

  auto& param = GeneratorPythia8Param::Instance();
  LOG(INFO) << "Init \'Pythia8\' generator with following parameters";
  LOG(INFO) << param;

  /** read configuration **/
  if (!param.config.empty()) {
    if (!mPythia.readFile(param.config, true)) {
      LOG(FATAL) << "Failed to init \'Pythia8\': problems with configuration file "
                 << param.config;
      return false;
    }
  }

  /** initialise **/
  if (!mPythia.init()) {
    LOG(FATAL) << "Failed to init \'Pythia8\': init returned with error";
    return false;
  }

  /** success **/
  return true;
}

/*****************************************************************/

Bool_t
  GeneratorPythia8::generateEvent()
{
  /** generate event **/

  return mPythia.next();
}

/*****************************************************************/

Bool_t
  GeneratorPythia8::importParticles()
{
  /** import particles **/

  TClonesArray& clonesParticles = *mParticles;
  clonesParticles.Clear();

  /* loop over particles */
  //  auto weight = mPythia.info.weight(); // TBD: use weights
  auto nParticles = mPythia.event.size();
  for (Int_t iparticle = 1; iparticle < nParticles; iparticle++) { // first particle is system
    auto particle = mPythia.event[iparticle];
    auto pdg = particle.id();
    auto st = particle.statusHepMC();
    auto px = particle.px();
    auto py = particle.py();
    auto pz = particle.pz();
    auto et = particle.e();
    auto vx = particle.xProd();
    auto vy = particle.yProd();
    auto vz = particle.zProd();
    auto vt = particle.tProd();
    auto m1 = particle.mother1();
    auto m2 = particle.mother2();
    auto d1 = particle.daughter1();
    auto d2 = particle.daughter2();
    new (clonesParticles[iparticle]) TParticle(pdg, st, m1, m2, d1, d2, px, py, pz, et, vx, vy, vz, vt);
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
