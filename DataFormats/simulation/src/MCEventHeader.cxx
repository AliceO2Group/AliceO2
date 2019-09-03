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

#include "SimulationDataFormat/MCEventHeader.h"
#include "FairRootManager.h"

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

void MCEventHeader::Reset()
{
  /** reset **/

  mEmbeddingFileName.clear();
  mEmbeddingEventIndex = 0;
  FairMCEventHeader::Reset();
}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

ClassImp(o2::dataformats::MCEventHeader);
