// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Generators/TriggerTGenerator.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

TriggerTGenerator::TriggerTGenerator() : Trigger() { /** default constructor **/}

/*****************************************************************/

TriggerTGenerator::~TriggerTGenerator() { /** default destructor **/}

/*****************************************************************/

Bool_t TriggerTGenerator::triggerEvent(TClonesArray* particles, TGenerator* generator)
{
  /** trigger event **/

  /** check active **/
  if (!isActive())
    return kFALSE;
  /* trigger */
  if (!isTriggered(particles, generator))
    return kFALSE;
  /* downscale */
  if (isDownscaled())
    return kFALSE;

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */
