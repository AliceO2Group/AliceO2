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

/// \author R+Preghenella - January 2020

#include "Generators/GeneratorPythia8.h"
#include "Generators/GeneratorPythia8Param.h"
#include "CommonUtils/ConfigurationMacroHelper.h"
#include <fairlogger/Logger.h>
#include "TParticle.h"
#include "TF1.h"
#include "TRandom.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "Pythia8/HIUserHooks.h"
#include "Pythia8Plugins/PowhegHooks.h"
#include "TSystem.h"
#include "ZDCBase/FragmentParam.h"
#include <CommonUtils/ConfigurationMacroHelper.h>
#include <filesystem>
#include <CommonUtils/FileSystemUtils.h>

#include <iostream>
#include <unordered_map>
#include <numeric>

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
  LOG(info) << "Instance \'Pythia8\' generator with following parameters";
  LOG(info) << param;

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
      LOG(info) << "Reading configuration from file: " << config;
      if (!mPythia.readFile(config, true)) {
        LOG(fatal) << "Failed to init \'Pythia8\': problems with configuration file "
                   << config;
        return false;
      }
    }
  }

  /** user hooks via configuration macro **/
  if (!mHooksFileName.empty()) {
    LOG(info) << "Applying \'Pythia8\' user hooks: " << mHooksFileName << " -> " << mHooksFuncName;
    auto hooks = o2::conf::GetFromMacro<Pythia8::UserHooks*>(mHooksFileName, mHooksFuncName, "Pythia8::UserHooks*", "pythia8_user_hooks");
    if (!hooks) {
      LOG(fatal) << "Failed to init \'Pythia8\': problem with user hooks configuration ";
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
  if (mPythia.settings.mode("Beams:frameType") == 4) {
    // Hook for POWHEG
    // Read in key POWHEG merging settings
    int vetoMode = mPythia.settings.mode("POWHEG:veto");
    int MPIvetoMode = mPythia.settings.mode("POWHEG:MPIveto");
    bool loadHooks = (vetoMode > 0 || MPIvetoMode > 0);
    // Add in user hooks for shower vetoing
    std::shared_ptr<Pythia8::PowhegHooks> powhegHooks;
    if (loadHooks) {
      // Set ISR and FSR to start at the kinematical limit
      if (vetoMode > 0) {
        mPythia.readString("SpaceShower:pTmaxMatch = 2");
        mPythia.readString("TimeShower:pTmaxMatch = 2");
      }
      // Set MPI to start at the kinematical limit
      if (MPIvetoMode > 0) {
        mPythia.readString("MultipartonInteractions:pTmaxMatch = 2");
      }
      powhegHooks = std::make_shared<Pythia8::PowhegHooks>();
      mPythia.setUserHooksPtr((Pythia8::UserHooksPtr)powhegHooks);
    }
  }
  /** initialise **/
  if (!mPythia.init()) {
    LOG(fatal) << "Failed to init \'Pythia8\': init returned with error";
    return false;
  }

  initUserFilterCallback();

  /** success **/
  return true;
}

/*****************************************************************/
void GeneratorPythia8::setUserHooks(Pythia8::UserHooks* hooks)
{
#if PYTHIA_VERSION_INTEGER < 8300
  mPythia.setUserHooksPtr(hooks);
#else
  mPythia.setUserHooksPtr(std::shared_ptr<Pythia8::UserHooks>(hooks));
#endif
}

/*****************************************************************/

Bool_t
  GeneratorPythia8::generateEvent()
{
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
void GeneratorPythia8::investigateRelatives(Pythia8::Event& event,
                                            const std::vector<int>& old2New,
                                            size_t index,
                                            std::vector<bool>& done,
                                            GetRelatives getter,
                                            SetRelatives setter,
                                            FirstLastRelative firstLast,
                                            const std::string& what,
                                            const std::string& ind)
{
  // Utility to find new index, or -1 if not found
  auto findNew = [old2New](size_t old) -> int {
    return old2New[old];
  };
  int newIdx = findNew(index);
  int hepmc = event[index].statusHepMC();

  LOG(debug) << ind
             << index << " -> "
             << newIdx << " (" << hepmc << ") ";
  if (done[index]) {
    LOG(debug) << ind << " already done";
    return;
  }

  // Our list of new relatives
  using IdList = std::pair<int, int>;
  constexpr int invalid = 0xFFFFFFF;
  IdList newRelatives = std::make_pair(invalid, -invalid);

  // Utility to add id
  auto addId = [](IdList& l, size_t id) {
    l.first = std::min(int(id), l.first);
    l.second = std::max(int(id), l.second);
  };

  // Get particle and relatives
  auto& particle = event[index];
  auto relatives = getter(particle);

  LOG(debug) << ind << " Check " << what << "s ["
             << std::setw(3) << firstLast(particle).first << ","
             << std::setw(3) << firstLast(particle).second << "] "
             << relatives.size();

  for (auto relativeIdx : relatives) {
    int newRelative = findNew(relativeIdx);
    if (newRelative >= 0) {
      // If this relative is to be kept, then append to list of new
      // relatives.
      LOG(debug) << ind << " "
                 << what << " "
                 << relativeIdx << " -> "
                 << newRelative << " to be kept" << std::endl;
      addId(newRelatives, newRelative);
      continue;
    }
    LOG(debug) << ind << " "
               << what << " "
               << relativeIdx << " not to be kept "
               << (done[relativeIdx] ? "already done" : "to be done")
               << std::endl;

    // Below is code for when the relative is not to be kept
    auto& relative = event[relativeIdx];
    if (not done[relativeIdx]) {
      // IF the relative hasn't been processed yet, do so now
      investigateRelatives(event,       // Event
                           old2New,     // Map from old to new
                           relativeIdx, // current particle index
                           done,        // cache flag
                           getter,      // get mother relatives
                           setter,      // set mother relatives
                           firstLast,   // get first and last
                           what,        // what we're looking at
                           ind + "  "); // Logging indent
    }

    // If this relative was already done, then get its relatives and
    // add them to the list of new relatives.
    auto grandRelatives = firstLast(relative);
    int grandRelative1 = grandRelatives.first;
    int grandRelative2 = grandRelatives.second;
    assert(grandRelative1 != invalid);
    assert(grandRelative2 != -invalid);
    if (grandRelative1 > 0) {
      addId(newRelatives, grandRelative1);
    }
    if (grandRelative2 > 0) {
      addId(newRelatives, grandRelative2);
    }
    LOG(debug) << ind << " "
               << what << " "
               << relativeIdx << " gave new relatives "
               << grandRelative1 << " -> " << grandRelative2;
  }
  LOG(debug) << ind << " Got "
             << (newRelatives.second - newRelatives.first + 1) << " new "
             << what << "s ";

  if (newRelatives.first != invalid) {
    // If the first relative is not invalid, then the second isn't
    // either (possibly the same though).
    int newRelative1 = newRelatives.first;
    int newRelative2 = newRelatives.second;
    setter(particle, newRelative1, newRelative2);
    LOG(debug) << ind << " " << what << "s: "
               << firstLast(particle).first << " ("
               << newRelative1 << "),"
               << firstLast(particle).second << " ("
               << newRelative2 << ")";

  } else {
    setter(particle, 0, 0);
  }
  done[index] = true;
}

/*****************************************************************/
void GeneratorPythia8::pruneEvent(Pythia8::Event& event, Select select)
{
  // Mapping from old to new index.
  std::vector<int> old2new(event.size(), -1);

  // Particle 0 is a system particle, and we will skip that in the
  // following.
  size_t newId = 0;

  // Loop over particles and store those we need
  for (size_t i = 1; i < event.size(); i++) {
    auto& particle = event[i];
    if (select(particle)) {
      ++newId;
      old2new[i] = newId;
    }
  }
  // Utility to find new index, or -1 if not found
  auto findNew = [old2new](size_t old) -> int {
    return old2new[old];
  };

  // First loop, investigate mothers - from the bottom
  auto getMothers = [](const Pythia8::Particle& particle) { return particle.motherList(); };
  auto setMothers = [](Pythia8::Particle& particle, int m1, int m2) { particle.mothers(m1, m2); };
  auto firstLastMothers = [](const Pythia8::Particle& particle) { return std::make_pair(particle.mother1(), particle.mother2()); };

  std::vector<bool> motherDone(event.size(), false);
  for (size_t i = 1; i < event.size(); ++i) {
    investigateRelatives(event,            // Event
                         old2new,          // Map from old to new
                         i,                // current particle index
                         motherDone,       // cache flag
                         getMothers,       // get mother relatives
                         setMothers,       // set mother relatives
                         firstLastMothers, // get first and last
                         "mother");        // what we're looking at
  }

  // Second loop, investigate daughters - from the top
  auto getDaughters = [](const Pythia8::Particle& particle) {
    // In case of |status|==13 (diffractive), we cannot use
    // Pythia8::Particle::daughterList as it will give more than
    // just the immediate daughters. In that cae, we do it
    // ourselves.
    if (std::abs(particle.status()) == 13) {
      int d1 = particle.daughter1();
      int d2 = particle.daughter2();
      if (d1 == 0 and d2 == 0) {
        return std::vector<int>();
      }
      if (d2 == 0) {
        return std::vector<int>{d1};
      }
      if (d2 > d1) {
        std::vector<int> ret(d2-d1+1);
        std::iota(ret.begin(), ret.end(), d1);
        return ret;
      }
      return std::vector<int>{d2,d1};
    }
    return particle.daughterList(); };
  auto setDaughters = [](Pythia8::Particle& particle, int d1, int d2) { particle.daughters(d1, d2); };
  auto firstLastDaughters = [](const Pythia8::Particle& particle) { return std::make_pair(particle.daughter1(), particle.daughter2()); };

  std::vector<bool> daughterDone(event.size(), false);
  for (size_t i = event.size() - 1; i > 0; --i) {
    investigateRelatives(event,              // Event
                         old2new,            // Map from old to new
                         i,                  // current particle index
                         daughterDone,       // cache flag
                         getDaughters,       // get mother relatives
                         setDaughters,       // set mother relatives
                         firstLastDaughters, // get first and last
                         "daughter");        // what we're looking at
  }

  // Make a pruned event
  Pythia8::Event pruned;
  pruned.init("Pruned event", &mPythia.particleData);
  pruned.reset();

  for (size_t i = 1; i < event.size(); i++) {
    int newIdx = findNew(i);
    if (newIdx < 0) {
      continue;
    }

    auto particle = event[i];
    int realIdx = pruned.append(particle);
    assert(realIdx == newIdx);
  }

  // We may have that two or more mothers share some daughters, but
  // that one or more mothers have more daughters than the other
  // mothers, and hence not all daughters point back to all mothers.
  // This can happen, for example, if a beam particle radiates
  // on-shell particles before an interaction with any daughters
  // from the other mothers.  Thus, we need to take care of that or
  // the event record will be corrupted.
  //
  // What we do is that for all particles, we look up the daughters.
  // Then for each daughter, we check the mothers of those
  // daughters.  If this list of mothers include other mothers than
  // the currently investigated mother, we must change the mothers
  // of the currently investigated daughters.
  using IdList = std::pair<int, int>;
  // Utility to add id
  auto addId = [](IdList& l, size_t id) {
    l.first = std::min(int(id), l.first);
    l.second = std::max(int(id), l.second);
  };
  constexpr int invalid = 0xFFFFFFF;

  std::vector<bool> shareDone(pruned.size(), false);
  for (size_t i = 1; i < pruned.size(); i++) {
    if (shareDone[i]) {
      continue;
    }

    auto& particle = pruned[i];
    auto daughters = particle.daughterList();
    IdList allDaughters = std::make_pair(invalid, -invalid);
    IdList allMothers = std::make_pair(invalid, -invalid);
    addId(allMothers, i);
    for (auto daughterIdx : daughters) {
      // Add this daughter to set of all daughters
      addId(allDaughters, daughterIdx);
      auto& daughter = pruned[daughterIdx];
      auto otherMothers = daughter.motherList();
      for (auto otherMotherIdx : otherMothers) {
        // Add this mother to set of all mothers.  That is, take all
        // mothers of the current daughter of the current particle
        // and store that.  In this way, we register mothers that
        // share a daughter with the current particle.
        addId(allMothers, otherMotherIdx);
        // We also need to take all the daughters of this shared
        // mother and reister those.
        auto& otherMother = pruned[otherMotherIdx];
        int otherDaughter1 = otherMother.daughter1();
        int otherDaughter2 = otherMother.daughter2();
        if (otherDaughter1 > 0) {
          addId(allDaughters, otherDaughter1);
        }
        if (otherDaughter2 > 0) {
          addId(allDaughters, otherDaughter2);
        }
      }
      // At this point, we have added all mothers of current
      // daughter, and all daughters of those mothers.
    }
    // At this point, we have all mothers that share daughters with
    // the current particle, and we have all of the daughters
    // too.
    //
    // We can now update the daughter information on all mothers
    int minDaughter = allDaughters.first;
    int maxDaughter = allDaughters.second;
    int minMother = allMothers.first;
    int maxMother = allMothers.second;
    if (minMother != invalid) {
      // If first mother isn't invalid, then second isn't either
      for (size_t motherIdx = minMother; motherIdx <= maxMother; //
           motherIdx++) {
        shareDone[motherIdx] = true;
        if (minDaughter == invalid) {
          pruned[motherIdx].daughters(0, 0);
        } else {
          pruned[motherIdx].daughters(minDaughter, maxDaughter);
        }
      }
    }
    if (minDaughter != invalid) {
      // If least mother isn't invalid, then largest mother will not
      // be invalid either.
      for (size_t daughterIdx = minDaughter; daughterIdx <= maxDaughter; //
           daughterIdx++) {
        if (minMother == invalid) {
          pruned[daughterIdx].mothers(0, 0);
        } else {
          pruned[daughterIdx].mothers(minMother, maxMother);
        }
      }
    }
  }
  LOG(info) << "Pythia event was pruned from " << event.size()
            << " to " << pruned.size() << " particles";
  // Assign our pruned event to the event passed in
  event = pruned;
}

/*****************************************************************/
void GeneratorPythia8::initUserFilterCallback()
{
  mUserFilterFcn = [](Pythia8::Particle const&) -> bool { return true; };

  auto& filter = GeneratorPythia8Param::Instance().particleFilter;
  if (filter.size() > 0) {
    LOG(info) << "Initializing the callback for user-based particle pruning " << filter;
    auto expandedFileName = o2::utils::expandShellVarsInFileName(filter);
    if (std::filesystem::exists(expandedFileName)) {
      // if the filter is in a file we will compile the hook on the fly
      mUserFilterFcn = o2::conf::GetFromMacro<UserFilterFcn>(expandedFileName, "filterPythia()", "o2::eventgen::GeneratorPythia8::UserFilterFcn", "o2mc_pythia8_userfilter_hook");
      LOG(info) << "Hook initialized from file " << expandedFileName;
    } else {
      // if it's not a file we interpret it as a C++ lambda string and JIT it directly;
      LOG(error) << "Did not find a file " << expandedFileName << " ; Will not execute hook";
    }
    mApplyPruning = true;
  }
}

/*****************************************************************/

Bool_t
  GeneratorPythia8::importParticles(Pythia8::Event& event)
{
  /** import particles **/

  // The right moment to filter out unwanted stuff (like parton-level
  // event information) Here, we aim to filter out everything before
  // hadronization with the motivation to reduce the size of the MC
  // event record in the AOD.

  std::function<bool(const Pythia8::Particle&)> partonSelect = [](const Pythia8::Particle&) { return true; };
  if (not GeneratorPythia8Param::Instance().includePartonEvent) {

    // Select pythia particles
    partonSelect = [](const Pythia8::Particle& particle) {
      switch (particle.statusHepMC()) {
        case 1: // Final st
        case 2: // Decayed
        case 4: // Beam
          return true;
      }
      // For example to keep diffractive particles
      // if (particle.id() == 9902210) return true;
      return false;
    };
    mApplyPruning = true;
  }

  if (mApplyPruning) {
    auto finalSelect = [partonSelect, this](const Pythia8::Particle& p) { return partonSelect(p) && mUserFilterFcn(p); };
    pruneEvent(event, finalSelect);
  }

  /* loop over particles */
  auto nParticles = event.size();
  for (Int_t iparticle = 1; iparticle < nParticles; iparticle++) {
    // first particle is system
    auto particle = event[iparticle];
    auto pdg = particle.id();
    auto st = o2::mcgenstatus::MCGenStatusEncoding(particle.statusHepMC(), //
                                                   particle.status())      //
                .fullEncoding;
    mParticles.push_back(TParticle(particle.id(),            // Particle type
                                   st,                       // status
                                   particle.mother1() - 1,   // first mother
                                   particle.mother2() - 1,   // second mother
                                   particle.daughter1() - 1, // first daughter
                                   particle.daughter2() - 1, // second daughter
                                   particle.px(),            // X-momentum
                                   particle.py(),            // Y-momentum
                                   particle.pz(),            // Z-momentum
                                   particle.e(),             // Energy
                                   particle.xProd(),         // Production X
                                   particle.yProd(),         // Production Y
                                   particle.zProd(),         // Production Z
                                   particle.tProd()));       // Production t
    mParticles.back().SetBit(ParticleStatus::kToBeDone,      //
                             particle.statusHepMC() == 1);
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/

void GeneratorPythia8::updateHeader(o2::dataformats::MCEventHeader* eventHeader)
{
  /** update header **/
  using Key = o2::dataformats::MCInfoKeys;

  eventHeader->putInfo<std::string>(Key::generator, "pythia8");
  eventHeader->putInfo<int>(Key::generatorVersion, PYTHIA_VERSION_INTEGER);
  eventHeader->putInfo<std::string>(Key::processName, mPythia.info.name());
  eventHeader->putInfo<int>(Key::processCode, mPythia.info.code());
  eventHeader->putInfo<float>(Key::weight, mPythia.info.weight());

  auto& info = mPythia.info;

  // Set PDF information
  eventHeader->putInfo<int>(Key::pdfParton1Id, info.id1pdf());
  eventHeader->putInfo<int>(Key::pdfParton2Id, info.id2pdf());
  eventHeader->putInfo<float>(Key::pdfX1, info.x1pdf());
  eventHeader->putInfo<float>(Key::pdfX2, info.x2pdf());
  eventHeader->putInfo<float>(Key::pdfScale, info.QFac());
  eventHeader->putInfo<float>(Key::pdfXF1, info.pdf1());
  eventHeader->putInfo<float>(Key::pdfXF2, info.pdf2());

  // Set cross section
  eventHeader->putInfo<float>(Key::xSection, info.sigmaGen() * 1e9);
  eventHeader->putInfo<float>(Key::xSectionError, info.sigmaErr() * 1e9);

  // Set weights (overrides cross-section for each weight)
  size_t iw = 0;
  auto xsecErr = info.weightContainerPtr->getTotalXsecErr();
  for (auto w : info.weightContainerPtr->getTotalXsec()) {
    std::string post = (iw == 0 ? "" : "_" + std::to_string(iw));
    eventHeader->putInfo<float>(Key::weight + post, info.weightValueByIndex(iw));
    eventHeader->putInfo<float>(Key::xSection + post, w * 1e9);
    eventHeader->putInfo<float>(Key::xSectionError + post, xsecErr[iw] * 1e9);
    iw++;
  }

#if PYTHIA_VERSION_INTEGER < 8300
  auto hiinfo = mPythia.info.hiinfo;
#else
  auto hiinfo = mPythia.info.hiInfo;
#endif

  if (hiinfo) {
    /** set impact parameter **/
    eventHeader->SetB(hiinfo->b());
    eventHeader->putInfo<double>(Key::impactParameter, hiinfo->b());
    auto bImp = hiinfo->b();
    /** set Ncoll, Npart and Nremn **/
    int nColl, nPart;
    int nPartProtonProj, nPartNeutronProj, nPartProtonTarg, nPartNeutronTarg;
    int nRemnProtonProj, nRemnNeutronProj, nRemnProtonTarg, nRemnNeutronTarg;
    int nFreeNeutronProj, nFreeProtonProj, nFreeNeutronTarg, nFreeProtonTarg;
    getNcoll(nColl);
    getNpart(nPart);
    getNpart(nPartProtonProj, nPartNeutronProj, nPartProtonTarg, nPartNeutronTarg);
    getNremn(nRemnProtonProj, nRemnNeutronProj, nRemnProtonTarg, nRemnNeutronTarg);
    getNfreeSpec(nFreeNeutronProj, nFreeProtonProj, nFreeNeutronTarg, nFreeProtonTarg);
    eventHeader->putInfo<int>(Key::nColl, nColl);
    // These are all non-HepMC3 fields - of limited use
    eventHeader->putInfo<int>("Npart", nPart);
    eventHeader->putInfo<int>("Npart_proj_p", nPartProtonProj);
    eventHeader->putInfo<int>("Npart_proj_n", nPartNeutronProj);
    eventHeader->putInfo<int>("Npart_targ_p", nPartProtonTarg);
    eventHeader->putInfo<int>("Npart_targ_n", nPartNeutronTarg);
    eventHeader->putInfo<int>("Nremn_proj_p", nRemnProtonProj);
    eventHeader->putInfo<int>("Nremn_proj_n", nRemnNeutronProj);
    eventHeader->putInfo<int>("Nremn_targ_p", nRemnProtonTarg);
    eventHeader->putInfo<int>("Nremn_targ_n", nRemnNeutronTarg);
    eventHeader->putInfo<int>("Nfree_proj_n", nFreeNeutronProj);
    eventHeader->putInfo<int>("Nfree_proj_p", nFreeProtonProj);
    eventHeader->putInfo<int>("Nfree_targ_n", nFreeNeutronTarg);
    eventHeader->putInfo<int>("Nfree_targ_p", nFreeProtonTarg);

    // --- HepMC3 conforming information ---
    // This is how the Pythia authors define Ncoll
    // eventHeader->putInfo<int>(Key::nColl,
    //                           hiinfo->nAbsProj() + hiinfo->nDiffProj() +
    //                           hiinfo->nAbsTarg() + hiinfo->nDiffTarg() -
    //                           hiiinfo->nCollND() - hiinfo->nCollDD());
    eventHeader->putInfo<int>(Key::nPartProjectile,
                              hiinfo->nAbsProj() + hiinfo->nDiffProj());
    eventHeader->putInfo<int>(Key::nPartTarget,
                              hiinfo->nAbsTarg() + hiinfo->nDiffTarg());
    eventHeader->putInfo<int>(Key::nCollHard, hiinfo->nCollNDTot());
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

void GeneratorPythia8::getNcoll(const Pythia8::Info& info, int& nColl)
{

  /** compute number of collisions from sub-collision information **/

#if PYTHIA_VERSION_INTEGER < 8300
  auto hiinfo = info.hiinfo;
#else
  auto hiinfo = info.hiInfo;
#endif

  // This is how the Pythia authors define Ncoll
  nColl = (hiinfo->nAbsProj() + hiinfo->nDiffProj() +
           hiinfo->nAbsTarg() + hiinfo->nDiffTarg() -
           hiinfo->nCollND() - hiinfo->nCollDD());
  nColl = 0;

  if (!hiinfo) {
    return;
  }

  // loop over sub-collisions
  auto scptr = hiinfo->subCollisionsPtr();
  for (auto sc : *scptr) {

    // wounded nucleon flag in projectile/target
    auto pW = sc.proj->status() == Pythia8::Nucleon::ABS; // according to C.Bierlich this should be == Nucleon::ABS
    auto tW = sc.targ->status() == Pythia8::Nucleon::ABS;

    // increase number of collisions if both are wounded
    if (pW && tW) {
      nColl++;
    }
  }
}

/*****************************************************************/

void GeneratorPythia8::getNpart(const Pythia8::Info& info, int& nPart)
{

  /** compute number of participants as the sum of all participants nucleons **/

  // This is how the Pythia authors calculate Npart
#if PYTHIA_VERSION_INTEGER < 8300
  auto hiinfo = info.hiinfo;
#else
  auto hiinfo = info.hiInfo;
#endif
  if (hiinfo) {
    nPart = (hiinfo->nAbsProj() + hiinfo->nDiffProj() +
             hiinfo->nAbsTarg() + hiinfo->nDiffTarg());
  }

  int nProtonProj, nNeutronProj, nProtonTarg, nNeutronTarg;
  getNpart(info, nProtonProj, nNeutronProj, nProtonTarg, nNeutronTarg);
  nPart = nProtonProj + nNeutronProj + nProtonTarg + nNeutronTarg;
}

/*****************************************************************/

void GeneratorPythia8::getNpart(const Pythia8::Info& info, int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg)
{

  /** compute number of participants from sub-collision information **/

#if PYTHIA_VERSION_INTEGER < 8300
  auto hiinfo = info.hiinfo;
#else
  auto hiinfo = info.hiInfo;
#endif

  nProtonProj = nNeutronProj = nProtonTarg = nNeutronTarg = 0;
  if (!hiinfo) {
    return;
  }

  // keep track of wounded nucleons
  std::vector<Pythia8::Nucleon*> projW;
  std::vector<Pythia8::Nucleon*> targW;

  // loop over sub-collisions
  auto scptr = hiinfo->subCollisionsPtr();
  for (auto sc : *scptr) {

    // wounded nucleon flag in projectile/target
    auto pW = sc.proj->status() == Pythia8::Nucleon::ABS || sc.proj->status() == Pythia8::Nucleon::DIFF; // according to C.Bierlich this should be == Nucleon::ABS || Nucleon::DIFF
    auto tW = sc.targ->status() == Pythia8::Nucleon::ABS || sc.targ->status() == Pythia8::Nucleon::DIFF;

    // increase number of wounded projectile nucleons if not yet in the wounded vector
    if (pW && std::find(projW.begin(), projW.end(), sc.proj) == projW.end()) {
      projW.push_back(sc.proj);
      if (sc.proj->id() == 2212) {
        nProtonProj++;
      } else if (sc.proj->id() == 2112) {
        nNeutronProj++;
      }
    }

    // increase number of wounded target nucleons if not yet in the wounded vector
    if (tW && std::find(targW.begin(), targW.end(), sc.targ) == targW.end()) {
      targW.push_back(sc.targ);
      if (sc.targ->id() == 2212) {
        nProtonTarg++;
      } else if (sc.targ->id() == 2112) {
        nNeutronTarg++;
      }
    }
  }
}

/*****************************************************************/

void GeneratorPythia8::getNremn(const Pythia8::Event& event, int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg)
{

  /** compute number of spectators from the nuclear remnant of the beams **/

  // reset
  nProtonProj = nNeutronProj = nProtonTarg = nNeutronTarg = 0;
  auto nNucRem = 0;

  // particle loop
  auto nparticles = event.size();
  for (int ipa = 0; ipa < nparticles; ++ipa) {
    const auto particle = event[ipa];
    auto pdg = particle.id();

    // nuclear remnants have pdg code = ±10LZZZAAA9
    if (pdg < 1000000000) {
      continue; // must be nucleus
    }
    if (pdg % 10 != 9) {
      continue; // first digit must be 9
    }
    nNucRem++;

    // extract A, Z and L from pdg code
    pdg /= 10;
    auto A = pdg % 1000;
    pdg /= 1000;
    auto Z = pdg % 1000;
    pdg /= 1000;
    auto L = pdg % 10;

    if (particle.pz() > 0.) {
      nProtonProj = Z;
      nNeutronProj = A - Z;
    }
    if (particle.pz() < 0.) {
      nProtonTarg = Z;
      nNeutronTarg = A - Z;
    }

  } // end of particle loop

  if (nNucRem > 2) {
    LOG(warning) << " GeneratorPythia8: found more than two nuclear remnants (weird)";
  }
}
/*****************************************************************/

/*****************************************************************/

void GeneratorPythia8::getNfreeSpec(const Pythia8::Info& info, int& nFreenProj, int& nFreepProj, int& nFreenTarg, int& nFreepTarg)
{
  /** compute number of free spectator nucleons for ZDC response **/

#if PYTHIA_VERSION_INTEGER < 8300
  auto hiinfo = info.hiinfo;
#else
  auto hiinfo = info.hiInfo;
#endif

  if (!hiinfo) {
    return;
  }

  double b = hiinfo->b();

  static o2::zdc::FragmentParam frag; // data-driven model to get free spectators given impact parameter

  TF1 const& fneutrons = frag.getfNeutrons();
  TF1 const& fsigman = frag.getsigmaNeutrons();
  TF1 const& fprotons = frag.getfProtons();
  TF1 const& fsigmap = frag.getsigmaProtons();

  // Calculating no. of free spectators from parametrization
  int nneu[2] = {0, 0};
  for (int i = 0; i < 2; i++) {
    float nave = fneutrons.Eval(b);
    float sigman = fsigman.Eval(b);
    float nfree = gRandom->Gaus(nave, 0.68 * sigman * nave);
    nneu[i] = (int)nfree;
    if (nave < 0 || nneu[i] < 0) {
      nneu[i] = 0;
    }
    if (nneu[i] > 126) {
      nneu[i] = 126;
    }
  }
  //
  int npro[2] = {0, 0};
  for (int i = 0; i < 2; i++) {
    float pave = fprotons.Eval(b);
    float sigmap = fsigman.Eval(b);
    float pfree = gRandom->Gaus(pave, 0.68 * sigmap * pave) / 0.7;
    npro[i] = (int)pfree;
    if (pave < 0 || npro[i] < 0) {
      npro[i] = 0;
    }
    if (npro[i] > 82) {
      npro[i] = 82;
    }
  }

  nFreenProj = nneu[0];
  nFreenTarg = nneu[1];
  nFreepProj = npro[0];
  nFreepTarg = npro[1];
  /*****************************************************************/
}

} /* namespace eventgen */
} /* namespace o2 */
