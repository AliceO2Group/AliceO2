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
#include "FairPrimaryGenerator.h"
#include "FairLogger.h"
#include <cmath>

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
  Generator::ReadEvent(FairPrimaryGenerator* primGen)
{
  /** read event **/

  /** generate event **/
  if (!generateEvent())
    return kFALSE;

  /** boost event **/
  if (fabs(mBoost) > 0.001)
    if (!boostEvent(mBoost))
      return kFALSE;

  /** add tracks **/
  if (!addTracks(primGen))
    return kFALSE;

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::Generator);
