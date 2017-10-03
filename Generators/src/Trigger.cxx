// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Generators/Trigger.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

Trigger::Trigger()
  : TNamed("ALICEo2", "ALICEo2 Generator Trigger"),
    mDownscale(1.),
    mNumberOfTimeSlots(1),
    mActiveTimeSlot(0),
    mInvertActive(kFALSE),
    mTimeSlot(0)
{
  /** default contructor **/
}

/*****************************************************************/

Trigger::~Trigger() { /** default destructor **/}

/*****************************************************************/

Bool_t Trigger::isActive()
{
  /** trigger event **/

  /** check active time slot **/
  if ((!mInvertActive && mTimeSlot != mActiveTimeSlot) || (mInvertActive && mTimeSlot == mActiveTimeSlot)) {
    mTimeSlot = (mTimeSlot + 1) % mNumberOfTimeSlots;
    return kFALSE;
  }
  mTimeSlot = (mTimeSlot + 1) % mNumberOfTimeSlots;

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::Trigger)
