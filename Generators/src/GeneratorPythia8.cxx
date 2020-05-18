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
#include "Generators/ConfigurationMacroHelper.h"
#include "FairLogger.h"
#include "TParticle.h"
#include "FairMCEventHeader.h"
#include "Pythia8/HIUserHooks.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

GeneratorPythia8::GeneratorPythia8() : Generator("ALICEo2", "ALICEo2 Pythia8 Generator")
{
  /** default constructor **/

  mInterface = reinterpret_cast<void*>(&mPythia);
  mInterfaceName = "pythia8";
}

/*****************************************************************/

GeneratorPythia8::GeneratorPythia8(const Char_t* name, const Char_t* title) : Generator(name, title)
{
  /** constructor **/

  mInterface = reinterpret_cast<void*>(&mPythia);
  mInterfaceName = "pythia8";
}

/*****************************************************************/

Bool_t GeneratorPythia8::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

  /** read configuration **/
  if (!mConfig.empty()) {
    if (!mPythia.readFile(mConfig, true)) {
      LOG(FATAL) << "Failed to init \'Pythia8\': problems with configuration file "
                 << mConfig;
      return false;
    }
  }

  /** user hooks via configuration macro **/
  if (!mHooksFileName.empty()) {
    auto hooks = GetFromMacro<Pythia8::UserHooks*>(mHooksFileName, mHooksFuncName, "Pythia8::UserHooks*", "pythia8_user_hooks");
    if (!hooks) {
      LOG(FATAL) << "Failed to init \'Pythia8\': problem with user hooks configuration ";
      return false;
    }
#if PYTHIA_VERSION_INTEGER < 8300
    mPythia.setUserHooksPtr(hooks);
#else
    mPythia.setUserHooksPtr(std::shared_ptr<Pythia8::UserHooks>(hooks));
#endif
  }

#if PYTHIA_VERSION_INTEGER < 8300
  /** [NOTE] The issue with large particle production vertex when running 
      Pythia8 heavy-ion model (Angantyr) is solved in Pythia 8.3 series.
      For discussions about this issue, please refer to this JIRA ticket
      https://alice.its.cern.ch/jira/browse/O2-1382.
      The code remains within preprocessor directives, both for reference
      and in case future use demands to roll back to Pythia 8.2 series. **/

  /** inhibit hadron decays **/
  mPythia.readString("HadronLevel:Decay off");
#endif

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

  /** generate event **/
  if (!mPythia.next())
    return false;

#if PYTHIA_VERSION_INTEGER < 8300
  /** [NOTE] The issue with large particle production vertex when running 
      Pythia8 heavy-ion model (Angantyr) is solved in Pythia 8.3 series.
      For discussions about this issue, please refer to this JIRA ticket
      https://alice.its.cern.ch/jira/browse/O2-1382.
      The code remains within preprocessor directives, both for reference
      and in case future use demands to roll back to Pythia 8.2 series. **/

  /** As we have inhibited all hadron decays before init,
      the event generation stops after hadronisation.
      We then pick all particles from here and force their
      production vertex to be (0,0,0,0).
      Afterwards we process the decays. **/

  /** force production vertices to (0,0,0,0) **/
  auto nParticles = mPythia.event.size();
  for (int iparticle = 0; iparticle < nParticles; iparticle++) {
    auto& aParticle = mPythia.event[iparticle];
    aParticle.xProd(0.);
    aParticle.yProd(0.);
    aParticle.zProd(0.);
    aParticle.tProd(0.);
  }

  /** proceed with decays **/
  if (!mPythia.moreDecays())
    return false;
#endif

  /** success **/
  return true;
}

/*****************************************************************/

Bool_t
  GeneratorPythia8::importParticles()
{
  /** import particles **/

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
    mParticles.push_back(TParticle(pdg, st, m1, m2, d1, d2, px, py, pz, et, vx, vy, vz, vt));
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/

void GeneratorPythia8::updateHeader(FairMCEventHeader* eventHeader)
{
  /** update header **/

#if PYTHIA_VERSION_INTEGER < 8300
  auto hiinfo = mPythia.info.hiinfo;
#else
  auto hiinfo = mPythia.info.hiInfo;
#endif

  /** set impact parameter if in heavy-ion mode **/
  if (hiinfo)
    eventHeader->SetB(hiinfo->b());
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
