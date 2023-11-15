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

#include <iostream>
#include <unordered_map>

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

  /** success **/
  return true;
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

Bool_t
  GeneratorPythia8::importParticles(Pythia8::Event const& event)
{
  /** import particles **/

  // the right moment to filter out unwanted stuff (like parton-level event information)

  std::unique_ptr<Pythia8::Event> hadronLevelEvent;
  auto eventToRead = &event;

  // The right moment to filter out unwanted stuff (like parton-level event information)
  // Here, we aim to filter out everything before hadronization with the motivation to reduce the
  // size of the MC event record in the AOD.
  if (!GeneratorPythia8Param::Instance().includePartonEvent) {

    // lambda performing the actual filtering
    auto getHadronLevelEvent = [](Pythia8::Event const& event, Pythia8::Event& hadronLevelEvent) {
      std::unordered_map<int, int> old_to_new;
      std::vector<Pythia8::Particle> filtered;
      // push the system particle
      filtered.push_back(event[0]);

      // Iterate over all particles and keep those that appear in hadronization phase
      // (should be mostly those with HepMC statuses 1 and 2)
      // we go from 1 since 0 is system as whole
      for (int i = 1; i < event.size(); ++i) {
        auto& p = event[i];
        if (p.statusHepMC() == 1 || p.statusHepMC() == 2) {
          filtered.push_back(p);
          old_to_new[i] = filtered.size() - 1;
        }
      }

      // helper lambda to lookup new index in filtered event - returns new id or -1 if not succesfull
      auto lookupNew = [&old_to_new](int oldid) {
        auto iter = old_to_new.find(oldid);
        if (iter == old_to_new.end()) {
          return -1;
        }
        return iter->second;
      };

      std::vector<int> childbuffer;

      // a lambda to check/assert size on children
      auto checkChildrenSize = [&childbuffer](int expected) {
        if (expected != childbuffer.size()) {
          LOG(error) << "Transcribed children list does not have expected size " << expected << " but " << childbuffer.size();
        }
      };

      // second pass to fix parent / children mappings
      for (int i = 1; i < filtered.size(); ++i) {
        auto& p = filtered[i];
        // get old daughters --> lookup their new position and fix
        auto originaldaughterids = p.daughterList();

        // this checks if all children have been copied over to filtered
        childbuffer.clear();
        for (auto& oldid : originaldaughterids) {
          auto newid = lookupNew(oldid);
          if (newid == -1) {
            LOG(error) << "Pythia8 remapping error - original index not known " << oldid;
          } else {
            childbuffer.push_back(newid);
          }
        }

        // fix children
        // analyse the cases (see Pythia8 documentation)
        auto d1 = p.daughter1();
        auto d2 = p.daughter2();
        if (d1 == 0 && d2 == 0) {
          // there is no offsprint --> nothing to do
          checkChildrenSize(0);
        } else if (d1 == d2 && d1 != 0) {
          // carbon copy ... should not happend here
          checkChildrenSize(1);
          p.daughters(childbuffer[0], childbuffer[0]);
        } else if (d1 > 0 && d2 == 0) {
          checkChildrenSize(1);
          p.daughters(childbuffer[0], 0);
        } else if (d2 != 0 && d2 > d1) {
          // multiple decay products ... adjacent in the event
          checkChildrenSize(d2 - d1 + 1);
          p.daughters(lookupNew(d1), lookupNew(d2));
        } else if (d2 != 0 && d2 < d1) {
          // 2 distinct products ... not adjacent to each other
          checkChildrenSize(2);
          p.daughters(lookupNew(d1), lookupNew(d2));
        }

        // fix mothers
        auto m1 = p.mother1();
        auto m2 = p.mother2();
        if (m1 == 0 && m2 == 0) {
          // nothing to be done
        } else if (m1 > 0 && m2 == m1) {
          // carbon copy
          auto tmp = lookupNew(m1);
          if (tmp != -1) {
            p.mothers(tmp, tmp);
          } else {
            // delete mother link since no longer available
            p.mothers(0, 0);
          }
        } else if (m1 > 0 && m2 == 0) {
          // the "normal" mother case, where it is meaningful to speak of one single mother to several products, in a shower or decay;
          auto tmp = lookupNew(m1);
          if (tmp != -1) {
            p.mothers(tmp, tmp);
          } else {
            // delete mother link since no longer available
            p.mothers(0, 0);
          }
        } else if (m1 < m2 && m1 > 0) {
          // mother1 < mother2, both > 0,
          // case for abs(status) = 81 - 86: primary hadrons produced from
          // the fragmentation of a string spanning the range from mother1 to mother2,
          // so that all partons in this range should be considered mothers;
          // and analogously for abs(status) = 101 - 106, the formation of R-hadrons;

          // here we simply delete the mothers
          p.mothers(0, 0);
          // verify that these shouldn't be in the list anyway
          if (lookupNew(m1) != -1 || lookupNew(m2) != -1) {
            LOG(warn) << "Indexing looks weird for primary hadron cases";
          }
        } else {
          LOG(warn) << "Unsupported / unexpected mother reindexing. Code needs more treatment";
        }
        // append this to the Pythia event
        hadronLevelEvent.append(p);
      }
    };

    hadronLevelEvent.reset(new Pythia8::Event);
    hadronLevelEvent->init("Hadron Level event record", &mPythia.particleData);

    hadronLevelEvent->reset();

    getHadronLevelEvent(event, *hadronLevelEvent);

    hadronLevelEvent->list();
    eventToRead = hadronLevelEvent.get();
    LOG(info) << "The pythia event has been reduced from size " << event.size()
              << " to " << hadronLevelEvent->size() << " by pre-hadronization pruning";
  }

  /* loop over particles */
  //  auto weight = mPythia.info.weight(); // TBD: use weights
  auto nParticles = eventToRead->size();
  for (Int_t iparticle = 1; iparticle < nParticles; iparticle++) { // first particle is system
    auto particle = (*eventToRead)[iparticle];
    auto pdg = particle.id();
    auto st = o2::mcgenstatus::MCGenStatusEncoding(particle.statusHepMC(), particle.status()).fullEncoding;
    auto px = particle.px();
    auto py = particle.py();
    auto pz = particle.pz();
    auto et = particle.e();
    auto vx = particle.xProd();
    auto vy = particle.yProd();
    auto vz = particle.zProd();
    auto vt = particle.tProd();
    auto m1 = particle.mother1() - 1;
    auto m2 = particle.mother2() - 1;
    auto d1 = particle.daughter1() - 1;
    auto d2 = particle.daughter2() - 1;
    mParticles.push_back(TParticle(pdg, st, m1, m2, d1, d2, px, py, pz, et, vx, vy, vz, vt));
    mParticles.back().SetBit(ParticleStatus::kToBeDone, particle.statusHepMC() == 1);
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
