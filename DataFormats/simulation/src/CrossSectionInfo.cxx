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

#include "SimulationDataFormat/CrossSectionInfo.h"
#include <iostream>

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

CrossSectionInfo::CrossSectionInfo()
  : GeneratorInfo(), mCrossSection(0.), mCrossSectionError(0.), mAcceptedEvents(0), mAttemptedEvents(0)
{
  /** default constructor **/
}

/*****************************************************************/

CrossSectionInfo::CrossSectionInfo(const CrossSectionInfo& rhs)
  : GeneratorInfo(rhs),
    mCrossSection(rhs.mCrossSection),
    mCrossSectionError(rhs.mCrossSectionError),
    mAcceptedEvents(rhs.mAcceptedEvents),
    mAttemptedEvents(rhs.mAttemptedEvents)
{
  /** copy constructor **/
}

/*****************************************************************/

CrossSectionInfo& CrossSectionInfo::operator=(const CrossSectionInfo& rhs)
{
  /** operator= **/

  if (this == &rhs)
    return *this;
  GeneratorInfo::operator=(rhs);
  mCrossSection = rhs.mCrossSection;
  mCrossSectionError = rhs.mCrossSectionError;
  mAcceptedEvents = rhs.mAcceptedEvents;
  mAttemptedEvents = rhs.mAttemptedEvents;
  return *this;
}

/*****************************************************************/

CrossSectionInfo::~CrossSectionInfo() { /** default destructor **/}

/*****************************************************************/

void CrossSectionInfo::reset()
{
  /** reset **/

  mCrossSection = 0.;
  mCrossSectionError = 0.;
  mAcceptedEvents = 0;
  mAttemptedEvents = 0;
}

/*****************************************************************/

void CrossSectionInfo::print() const
{
  /** print **/

  std::cout << ">>> sigma: " << mCrossSection << " +- " << mCrossSectionError << " (pb)"
            << " | accepted / attempted: " << mAcceptedEvents << " / " << mAttemptedEvents << std::endl;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */
