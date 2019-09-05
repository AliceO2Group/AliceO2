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

#ifndef ALICEO2_EVENTGEN_TRIGGERPARTICLE_H_
#define ALICEO2_EVENTGEN_TRIGGERPARTICLE_H_

#include "Generators/Trigger.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class TriggerParticle : public Trigger
{

 public:
  /** default constructor **/
  TriggerParticle() = default;
  /** destructor **/
  ~TriggerParticle() override = default;

  /** methods to override **/
  Bool_t fired(TClonesArray* particles) override;

  /** setters **/
  void setPDG(Int_t val) { mPDG = val; };
  void setPtRange(Double_t min, Double_t max)
  {
    mPtMin = min;
    mPtMax = max;
  };
  void setEtaRange(Double_t min, Double_t max)
  {
    mEtaMin = min;
    mEtaMax = max;
  };
  void setPhiRange(Double_t min, Double_t max)
  {
    mPhiMin = min;
    mPhiMax = max;
  };
  void setYRange(Double_t min, Double_t max)
  {
    mYMin = min;
    mYMax = max;
  };

 protected:
  /** copy constructor **/
  TriggerParticle(const TriggerParticle&);
  /** operator= **/
  TriggerParticle& operator=(const TriggerParticle&);

  /** trigger particle selection **/
  Int_t mPDG = 0;
  Double_t mPtMin = 0.;
  Double_t mPtMax = 1.e6;
  Double_t mEtaMin = -1.e6;
  Double_t mEtaMax = 1.e6;
  Double_t mPhiMin = -1.e6;
  Double_t mPhiMax = 1.e6;
  Double_t mYMin = -1.e6;
  Double_t mYMax = 1.e6;

  ClassDefOverride(TriggerParticle, 1);

}; /** class TriggerParticle **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_TRIGGERPARTICLE_H_ */
