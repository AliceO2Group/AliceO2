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

#include "Generators/Generator.h"
#include "Generators/Trigger.h"
#include "Generators/PrimaryGenerator.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "FairPrimaryGenerator.h"
#include <fairlogger/Logger.h>
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
    mReadEventCounter++;

    /** clear particle vector **/
    mParticles.clear();

    /** reset the sub-generator ID **/
    mSubGeneratorId = -1;

    /** generate event **/
    if (!generateEvent()) {
      return kFALSE;
    }

    /** import particles **/
    if (!importParticles()) {
      return kFALSE;
    }

    if (mSubGeneratorsIdToDesc.empty() && mSubGeneratorId > -1) {
      return kFALSE;
    }

    if (!mSubGeneratorsIdToDesc.empty() && mSubGeneratorId < 0) {
      return kFALSE;
    }

    /** trigger event **/
    if (triggerEvent()) {
      mTriggerOkHook(mParticles, mReadEventCounter);
      break;
    } else {
      mTriggerFalseHook(mParticles, mReadEventCounter);
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
    LOG(fatal) << "MC event header is not a 'o2::dataformats::MCEventHeader' object";
    return kFALSE;
  }
  updateHeader(o2header);
  updateSubGeneratorInformation(o2header);

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
    LOG(fatal) << "PrimaryGenerator is not a o2::eventgen::PrimaryGenerator";
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
                        particle.TestBit(ParticleStatus::kToBeDone),
                        particle.Energy() * mEnergyUnit,
                        particle.T() * mTimeUnit,
                        particle.GetWeight(),
                        (TMCProcess)particle.GetUniqueID(),
                        particle.GetStatusCode()); // generator status information passed as status code field
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

void Generator::addSubGenerator(int subGeneratorId, std::string const& subGeneratorDescription)
{
  if (mSubGeneratorId < 0) {
    LOG(fatal) << "Sub-generator IDs must be >= 0, instead, passed value is " << subGeneratorId;
  }
  mSubGeneratorsIdToDesc.insert({subGeneratorId, subGeneratorDescription});
}

/*****************************************************************/

void Generator::updateSubGeneratorInformation(o2::dataformats::MCEventHeader* header) const
{
  if (mSubGeneratorId < 0) {
    return;
  }
  header->putInfo<int>(o2::mcgenid::GeneratorProperty::SUBGENERATORID, mSubGeneratorId);
  header->putInfo<std::unordered_map<int, std::string>>(o2::mcgenid::GeneratorProperty::SUBGENERATORDESCRIPTIONMAP, mSubGeneratorsIdToDesc);
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::Generator);
