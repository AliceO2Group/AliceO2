// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2017

#include "Generators/PrimaryGenerator.h"
#include "Generators/InteractionDiamondParam.h"
#include "FairLogger.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/

Bool_t PrimaryGenerator::Init()
{
  /** init **/

  /** retrieve and set interaction diamond **/
  auto diamond = InteractionDiamondParam::Instance();
  setInteractionDiamond(diamond.position, diamond.width);

  /** base class init **/
  return FairPrimaryGenerator::Init();
}

/*****************************************************************/

void PrimaryGenerator::setInteractionDiamond(const Double_t* xyz, const Double_t* sigmaxyz)
{
  /** set interaction diamond **/

  LOG(INFO) << "Setting interaction diamond: position = {"
            << xyz[0] << "," << xyz[1] << "," << xyz[2] << "} cm";
  LOG(INFO) << "Setting interaction diamond: width = {"
            << sigmaxyz[0] << "," << sigmaxyz[1] << "," << sigmaxyz[2] << "} cm";
  SetBeam(xyz[0], xyz[1], sigmaxyz[0], sigmaxyz[1]);
  SetTarget(xyz[2], sigmaxyz[2]);
  SmearVertexXY(false);
  SmearVertexZ(false);
  SmearGausVertexXY(true);
  SmearGausVertexZ(true);
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::PrimaryGenerator)
