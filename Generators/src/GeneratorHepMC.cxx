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

/// \author R+Preghenella - August 2017

#include "Generators/GeneratorHepMC.h"
#include "Generators/GeneratorHepMCParam.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/ReaderAsciiHepMC2.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/FourVector.h"
#include "TParticle.h"
#include "TSystem.h"

#include "FairLogger.h"
#include "FairPrimaryGenerator.h"
#include <cmath>

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

GeneratorHepMC::GeneratorHepMC()
  : Generator("ALICEo2", "ALICEo2 HepMC Generator"), mStream(), mFileName(), mVersion(3), mReader(nullptr), mEvent(nullptr)
{
  /** default constructor **/

  mEvent = new HepMC3::GenEvent();
  mInterface = reinterpret_cast<void*>(mEvent);
  mInterfaceName = "hepmc";
}

/*****************************************************************/

GeneratorHepMC::GeneratorHepMC(const Char_t* name, const Char_t* title)
  : Generator(name, title), mStream(), mFileName(), mVersion(3), mReader(nullptr), mEvent(nullptr)
{
  /** constructor **/

  mEvent = new HepMC3::GenEvent();
  mInterface = reinterpret_cast<void*>(mEvent);
  mInterfaceName = "hepmc";
}

/*****************************************************************/

GeneratorHepMC::~GeneratorHepMC()
{
  /** default destructor **/

  if (mStream.is_open()) {
    mStream.close();
  }
  if (mReader) {
    mReader->close();
    delete mReader;
  }
  if (mEvent) {
    delete mEvent;
  }
}

/*****************************************************************/

Bool_t GeneratorHepMC::generateEvent()
{
  /** generate event **/

  /** clear and read event **/
  mEvent->clear();
  mReader->read_event(*mEvent);
  if (mReader->failed()) {
    return kFALSE;
  }
  /** set units to desired output **/
  mEvent->set_units(HepMC3::Units::GEV, HepMC3::Units::MM);

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t GeneratorHepMC::importParticles()
{
  /** import particles **/

  /** loop over particles **/
  auto particles = mEvent->particles();
  for (int i = 0; i < particles.size(); ++i) {

    /** get particle information **/
    auto particle = particles.at(i);
    auto pdg = particle->pid();
    auto st = particle->status();
    auto momentum = particle->momentum();
    auto vertex = particle->production_vertex()->position();
    auto parents = particle->parents();   // less efficient than via vertex
    auto children = particle->children(); // less efficient than via vertex

    /** get momentum information **/
    auto px = momentum.x();
    auto py = momentum.y();
    auto pz = momentum.z();
    auto et = momentum.t();

    /** get vertex information **/
    auto vx = vertex.x();
    auto vy = vertex.y();
    auto vz = vertex.z();
    auto vt = vertex.t();

    /** get mother information **/
    auto m1 = parents.empty() ? -1 : parents.front()->id() - 1;
    auto m2 = parents.empty() ? -1 : parents.back()->id() - 1;

    /** get daughter information **/
    auto d1 = children.empty() ? -1 : children.front()->id() - 1;
    auto d2 = children.empty() ? -1 : children.back()->id() - 1;

    /** add to particle vector **/
    mParticles.push_back(TParticle(pdg, st, m1, m2, d1, d2, px, py, pz, et, vx, vy, vz, vt));

  } /** end of loop over particles **/

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t GeneratorHepMC::Init()
{
  /** init **/

  /** init base class **/
  Generator::Init();

  /** open file **/
  std::string filename = gSystem->ExpandPathName(mFileName.c_str());
  mStream.open(filename);
  if (!mStream.is_open()) {
    LOG(fatal) << "Cannot open input file: " << filename << std::endl;
    return kFALSE;
  }

  /** create reader according to HepMC version **/
  switch (mVersion) {
    case 2:
      mStream.close();
      mReader = new HepMC3::ReaderAsciiHepMC2(filename);
      break;
    case 3:
      mReader = new HepMC3::ReaderAscii(mStream);
      break;
    default:
      LOG(fatal) << "Unsupported HepMC version: " << mVersion << std::endl;
      return kFALSE;
  }

  // skip events at the beginning
  if (!mReader->failed()) {
    LOGF(info, "%i events to skip.", mEventsToSkip);
    for (auto ind = 0; ind < mEventsToSkip; ind++) {
      if (!generateEvent()) {
        LOGF(error, "The file %s only contains %i events!", mFileName, ind);
        break;
      }
    }
  }

  /** success **/
  return !mReader->failed();
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
