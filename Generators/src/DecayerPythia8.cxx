// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2020

#include "Generators/DecayerPythia8.h"
#include "Generators/DecayerPythia8Param.h"
#include "FairLogger.h"
#include "TLorentzVector.h"
#include "TClonesArray.h"
#include "TParticle.h"
#include "TSystem.h"

#include <iostream>

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

void DecayerPythia8::Init()
{

  /** switch off process level **/
  mPythia.readString("ProcessLevel:all off");

  /** config **/
  auto& param = DecayerPythia8Param::Instance();
  LOG(INFO) << "Init \'DecayerPythia8\' with following parameters";
  LOG(INFO) << param;
  if (!param.config.empty()) {
    std::stringstream ss(param.config);
    std::string config;
    while (getline(ss, config, ' ')) {
      config = gSystem->ExpandPathName(config.c_str());
      if (!mPythia.readFile(config, true)) {
        LOG(FATAL) << "Failed to init \'DecayerPythia8\': problems with configuration file "
                   << config;
        return;
      }
    }
  }

  /** initialise **/
  if (!mPythia.init()) {
    LOG(FATAL) << "Failed to init \'DecayerPythia8\': init returned with error";
    return;
  }
}

/*****************************************************************/

void DecayerPythia8::Decay(Int_t pdg, TLorentzVector* lv)
{
  auto mayDecay = mPythia.particleData.mayDecay(pdg); // save mayDecay status
  mPythia.particleData.mayDecay(pdg, true);           // force decay

  mPythia.event.clear();
  mPythia.event.append(pdg, 11, 0, 0, lv->Px(), lv->Py(), lv->Pz(), lv->E(), lv->M());
  mPythia.moreDecays();

  mPythia.particleData.mayDecay(pdg, mayDecay); // restore mayDecay status
}

/*****************************************************************/

Int_t DecayerPythia8::ImportParticles(TClonesArray* particles)
{
  TClonesArray& ca = *particles;
  ca.Clear();

  auto nParticles = mPythia.event.size();
  for (Int_t iparticle = 1; iparticle < nParticles; iparticle++) {
    //  Do not import the decayed particle - start loop from 1
    auto particle = mPythia.event[iparticle];
    auto pdg = particle.id();
    auto st = particle.statusHepMC();
    auto px = particle.px();
    auto py = particle.py();
    auto pz = particle.pz();
    auto et = particle.e();
    auto vx = particle.xProd() * 0.1;
    auto vy = particle.yProd() * 0.1;
    auto vz = particle.zProd() * 0.1;
    auto vt = particle.tProd() * 3.3356410e-12;
    auto m1 = particle.mother1();
    auto m2 = particle.mother2();
    auto d1 = particle.daughter1();
    auto d2 = particle.daughter2();

    new (ca[iparticle - 1]) TParticle(pdg, st, m1, m2, d1, d2, px, py, pz, et, vx, vy, vz, vt);
  }

  return ca.GetEntries();
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::DecayerPythia8);
