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

#include "SimulationDataFormat/HeavyIonInfo.h"
#include <iostream>

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

HeavyIonInfo::HeavyIonInfo()
  : GeneratorInfo(),
    mNcollHard(0),
    mNpartProj(0),
    mNpartTarg(0),
    mNcoll(0),
    mNspecNeut(0),
    mNspecProt(0),
    mImpactParameter(0.),
    mEventPlaneAngle(0.),
    mEccentricity(0.),
    mSigmaNN(0.),
    mCentrality(0.)
{
  /** default constructor **/
}

/*****************************************************************/

HeavyIonInfo::HeavyIonInfo(const HeavyIonInfo& rhs)
  : GeneratorInfo(rhs),
    mNcollHard(rhs.mNcollHard),
    mNpartProj(rhs.mNpartProj),
    mNpartTarg(rhs.mNpartTarg),
    mNcoll(rhs.mNcoll),
    mNspecNeut(rhs.mNspecNeut),
    mNspecProt(rhs.mNspecProt),
    mImpactParameter(rhs.mImpactParameter),
    mEventPlaneAngle(rhs.mEventPlaneAngle),
    mEccentricity(rhs.mEccentricity),
    mSigmaNN(rhs.mSigmaNN),
    mCentrality(rhs.mCentrality)
{
  /** copy constructor **/
}

/*****************************************************************/

HeavyIonInfo& HeavyIonInfo::operator=(const HeavyIonInfo& rhs)
{
  /** operator= **/

  if (this == &rhs)
    return *this;
  GeneratorInfo::operator=(rhs);
  mNcollHard = rhs.mNcollHard;
  mNpartProj = rhs.mNpartProj;
  mNpartTarg = rhs.mNpartTarg;
  mNcoll = rhs.mNcoll;
  mNspecNeut = rhs.mNspecNeut;
  mNspecProt = rhs.mNspecProt;
  mImpactParameter = rhs.mImpactParameter;
  mEventPlaneAngle = rhs.mEventPlaneAngle;
  mEccentricity = rhs.mEccentricity;
  mSigmaNN = rhs.mSigmaNN;
  mCentrality = rhs.mCentrality;
  return *this;
}

/*****************************************************************/

HeavyIonInfo::~HeavyIonInfo() { /** default destructor **/}

/*****************************************************************/

void HeavyIonInfo::Reset()
{
  /** reset **/

  mNcollHard = 0;
  mNpartProj = 0;
  mNpartTarg = 0;
  mNcoll = 0;
  mNspecNeut = 0;
  mNspecProt = 0;
  mImpactParameter = 0.;
  mEventPlaneAngle = 0.;
  mEccentricity = 0.;
  mSigmaNN = 0.;
  mCentrality = 0.;
}

/*****************************************************************/

void HeavyIonInfo::Print(Option_t* opt) const
{
  /** print **/

  std::cout << ">>> Ncoll: " << mNcoll << " | Npart: " << mNpartProj + mNpartTarg << " | b: " << mImpactParameter
            << " (fm)"
            << " | cent: " << mCentrality << " (\%)" << std::endl;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

ClassImp(o2::dataformats::HeavyIonInfo)
