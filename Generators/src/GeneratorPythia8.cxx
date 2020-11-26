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
#include "Generators/GeneratorPythia8Param.h"
#include "CommonUtils/ConfigurationMacroHelper.h"
#include "FairLogger.h"
#include "TParticle.h"
#include "FairMCEventHeader.h"
#include "Pythia8/HIUserHooks.h"
#include "TSystem.h"

#include <iostream>

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

  auto& param = GeneratorPythia8Param::Instance();
  LOG(INFO) << "Instance \'Pythia8\' generator with following parameters";
  LOG(INFO) << param;

  setConfig(param.config);
  setHooksFileName(param.hooksFileName);
  setHooksFuncName(param.hooksFuncName);
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
    std::stringstream ss(mConfig);
    std::string config;
    while (getline(ss, config, ' ')) {
      config = gSystem->ExpandPathName(config.c_str());
      LOG(INFO) << "Reading configuration from file: " << config;
      if (!mPythia.readFile(config, true)) {
        LOG(FATAL) << "Failed to init \'Pythia8\': problems with configuration file "
                   << config;
        return false;
      }
    }
  }

  /** user hooks via configuration macro **/
  if (!mHooksFileName.empty()) {
    LOG(INFO) << "Applying \'Pythia8\' user hooks: " << mHooksFileName << " -> " << mHooksFuncName;
    auto hooks = o2::conf::GetFromMacro<Pythia8::UserHooks*>(mHooksFileName, mHooksFuncName, "Pythia8::UserHooks*", "pythia8_user_hooks");
    if (!hooks) {
      LOG(FATAL) << "Failed to init \'Pythia8\': problem with user hooks configuration ";
      return false;
    }
    setUserHooks(hooks);
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
  if (!mPythia.next()) {
    return false;
  }

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
  GeneratorPythia8::importParticles(Pythia8::Event& event)
{
  /** import particles **/

  /* loop over particles */
  //  auto weight = mPythia.info.weight(); // TBD: use weights
  auto nParticles = event.size();
  for (Int_t iparticle = 0; iparticle < nParticles; iparticle++) { // first particle is system
    auto particle = event[iparticle];
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
  if (hiinfo) {
    eventHeader->SetB(hiinfo->b());
  }
}

/*****************************************************************/

void GeneratorPythia8::selectFromAncestor(int ancestor, Pythia8::Event& inputEvent, Pythia8::Event& outputEvent)
{

  /** select from ancestor
      fills the output event with all particles related to
      an ancestor of the input event **/

  // recursive selection via lambda function
  std::set<int> selected;
  std::function<void(int)> select;
  select = [&](int i) {
    selected.insert(i);
    auto dl = inputEvent[i].daughterList();
    for (auto j : dl) {
      select(j);
    }
  };
  select(ancestor);

  // map selected particle index to output index
  std::map<int, int> indexMap;
  int index = outputEvent.size();
  for (auto i : selected) {
    indexMap[i] = index++;
  }

  // adjust mother/daughter indices and append to output event
  for (auto i : selected) {
    auto p = mPythia.event[i];
    auto m1 = indexMap[p.mother1()];
    auto m2 = indexMap[p.mother2()];
    auto d1 = indexMap[p.daughter1()];
    auto d2 = indexMap[p.daughter2()];
    p.mothers(m1, m2);
    p.daughters(d1, d2);

    outputEvent.append(p);
  }
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
