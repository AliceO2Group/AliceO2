// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - September 2019

#include "Generators/TriggerParticle.h"
#include "TClonesArray.h"
#include "TParticle.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

Bool_t
  TriggerParticle::fired(TClonesArray* particles)
{
  /** fired **/

  /** loop over particles **/
  Int_t nParticles = particles->GetEntries();
  TParticle* particle = nullptr;
  for (Int_t iparticle = 0; iparticle < nParticles; iparticle++) {
    particle = (TParticle*)particles->At(iparticle);
    if (!particle)
      continue;

    /** check PDG **/
    auto pdg = particle->GetPdgCode();
    if (pdg != mPDG)
      continue;

    /** check pt **/
    auto pt = particle->Pt();
    if (pt < mPtMin || pt > mPtMax)
      continue;

    /** check eta **/
    auto eta = particle->Eta();
    if (eta < mEtaMin || eta > mEtaMax)
      continue;

    /** check phi **/
    auto phi = particle->Phi();
    if (phi < mPhiMin || phi > mPhiMax)
      continue;

    /** check rapidity **/
    auto y = particle->Y();
    if (y < mYMin || phi > mYMax)
      continue;

    /** success **/
    return kTRUE;
  }

  /** failure **/
  return kFALSE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::TriggerParticle);
