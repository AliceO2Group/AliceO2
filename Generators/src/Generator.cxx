// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#include "Generators/Generator.h"
#include "Generators/Trigger.h"
#include "Generators/PrimaryGenerator.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "FairPrimaryGenerator.h"
#include "FairLogger.h"
#include <cmath>
#include "TClonesArray.h"
#include "TParticle.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

Generator::Generator() : FairGenerator("ALICEo2", "ALICEo2 Generator"),
                         mBoost(0.)
{
  /** default constructor **/

}

/*****************************************************************/

Generator::Generator(const Char_t* name, const Char_t* title) : FairGenerator(name, title),
                                                                mBoost(0.)
{
  /** constructor **/

}

/*****************************************************************/

Bool_t
  Generator::Init()
{
  /** init **/

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::ReadEvent(FairPrimaryGenerator* primGen)
{
  /** read event **/

  /** endless generate-and-trigger loop **/
  while (true) {

    /** clear particle vector **/
    mParticles.clear();

    /** generate event **/
    if (!generateEvent()) {
      return kFALSE;
    }

    /** import particles **/
    if (!importParticles()) {
      return kFALSE;
    }

    /** trigger event **/
    if (triggerEvent()) {
      break;
    }
  }

  /** add tracks **/
  if (!addTracks(primGen)) {
    return kFALSE;
  }

  /** update header **/
  auto header = primGen->GetEvent();
  auto o2header = dynamic_cast<o2::dataformats::MCEventHeader*>(header);
  if (!header) {
    LOG(FATAL) << "MC event header is not a 'o2::dataformats::MCEventHeader' object";
    return kFALSE;
  }
  updateHeader(o2header);

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::addTracks(FairPrimaryGenerator* primGen)
{
  /** add tracks **/

  auto o2primGen = dynamic_cast<PrimaryGenerator*>(primGen);
  if (!o2primGen) {
    LOG(FATAL) << "PrimaryGenerator is not a o2::eventgen::PrimaryGenerator";
    return kFALSE;
  }

  /** loop over particles **/
  for (const auto& particle : mParticles) {
    o2primGen->AddTrack(particle.GetPdgCode(),
                        particle.Px() * mMomentumUnit,
                        particle.Py() * mMomentumUnit,
                        particle.Pz() * mMomentumUnit,
                        particle.Vx() * mPositionUnit,
                        particle.Vy() * mPositionUnit,
                        particle.Vz() * mPositionUnit,
                        particle.GetMother(0),
                        particle.GetMother(1),
                        particle.GetDaughter(0),
                        particle.GetDaughter(1),
                        particle.GetStatusCode() == 1,
                        particle.Energy() * mEnergyUnit,
                        particle.T() * mTimeUnit,
                        particle.GetWeight());
  }

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::boostEvent()
{
  /** boost event **/

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t
  Generator::triggerEvent()
{
  /** trigger event **/

  /** check trigger presence **/
  if (mTriggers.size() == 0 && mDeepTriggers.size() == 0) {
    return kTRUE;
  }

  /** check trigger mode **/
  Bool_t triggered;
  if (mTriggerMode == kTriggerOFF) {
    return kTRUE;
  } else if (mTriggerMode == kTriggerOR) {
    triggered = kFALSE;
  } else if (mTriggerMode == kTriggerAND) {
    triggered = kTRUE;
  } else {
    return kTRUE;
  }

  /** loop over triggers **/
  for (const auto& trigger : mTriggers) {
    auto retval = trigger(mParticles);
    if (mTriggerMode == kTriggerOR) {
      triggered |= retval;
    }
    if (mTriggerMode == kTriggerAND) {
      triggered &= retval;
    }
  }

  /** loop over deep triggers **/
  for (const auto& trigger : mDeepTriggers) {
    auto retval = trigger(mInterface, mInterfaceName);
    if (mTriggerMode == kTriggerOR) {
      triggered |= retval;
    }
    if (mTriggerMode == kTriggerAND) {
      triggered &= retval;
    }
  }

  /** return **/
  return triggered;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::Generator);
