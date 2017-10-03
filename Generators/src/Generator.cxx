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
#include "Generators/PrimaryGenerator.h"
#include "Generators/Trigger.h"
#include "SimulationDataFormat/GeneratorHeader.h"
#include "FairPrimaryGenerator.h"
#include "FairLogger.h"
#include <cmath>

using o2::dataformats::GeneratorHeader;

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

Generator::Generator()
  : FairGenerator("ALICEo2", "ALICEo2 Generator"),
    mTriggerMode(kTriggerOFF),
    mMaxTriggerAttempts(100000),
    mTriggers(),
    mBoost(0.),
    mHeader(new GeneratorHeader())
{
  /** default constructor **/
}

/*****************************************************************/

Generator::Generator(const Char_t* name, const Char_t* title)
  : FairGenerator(name, title),
    mTriggerMode(kTriggerOFF),
    mMaxTriggerAttempts(100000),
    mTriggers(),
    mBoost(0.),
    mHeader(new GeneratorHeader(name))
{
  /** constructor **/
}

/*****************************************************************/

Generator::~Generator()
{
  /** default destructor **/

  for (auto& trigger : mTriggers)
    delete trigger;
  if (mHeader)
    delete mHeader;
}

/*****************************************************************/

Bool_t Generator::ReadEvent(FairPrimaryGenerator* primGen)
{
  /** read event **/

  /** reset header **/
  mHeader->Reset();

  /** trigger loop **/
  Int_t nAttempts = 0;
  do {

    /** check attempts **/
    nAttempts++;
    if (nAttempts % 1000 == 0)
      LOG(WARNING) << "Large number of trigger attempts: " << nAttempts << std::endl;
    else if (nAttempts > mMaxTriggerAttempts) {
      LOG(ERROR) << "Maximum number of trigger attempts exceeded: " << mMaxTriggerAttempts << std::endl;
      return kFALSE;
    }

    /** generate event **/
    if (!generateEvent())
      return kFALSE;

    /** boost event **/
    if (!boostEvent(mBoost))
      return kFALSE;

  } while (!triggerEvent()); /** end of trigger loop **/

  /** add tracks **/
  if (!addTracks(primGen))
    return kFALSE;

  /** setup header **/
  mHeader->setNumberOfAttempts(nAttempts);

  /** add header **/
  auto o2primGen = dynamic_cast<PrimaryGenerator*>(primGen);
  if (o2primGen && !addHeader(o2primGen))
    return kFALSE;

  /** success **/
  return kTRUE;
}

/*****************************************************************/

Bool_t Generator::triggerEvent() const
{
  /** trigger event **/

  auto triggered = kTRUE;
  if (mTriggers.size() == 0)
    return kTRUE;
  else if (mTriggerMode == kTriggerOFF)
    return kTRUE;
  else if (mTriggerMode == kTriggerOR)
    triggered = kFALSE;
  else if (mTriggerMode == kTriggerAND)
    triggered = kTRUE;
  else
    return kTRUE;

  /** loop over triggers **/
  for (const auto& trigger : mTriggers) {
    auto retval = triggerFired(trigger);
    if (mTriggerMode == kTriggerOR)
      triggered |= retval;
    if (mTriggerMode == kTriggerAND)
      triggered &= retval;
  } /** end of loop over triggers **/

  /** success **/
  return triggered;
}

/*****************************************************************/

Bool_t Generator::addHeader(PrimaryGenerator* primGen) const
{
  /** add header **/

  primGen->addHeader(mHeader);

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::Generator)
