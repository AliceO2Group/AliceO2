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

#ifndef ALICEO2_EVENTGEN_TRIGGER_H_
#define ALICEO2_EVENTGEN_TRIGGER_H_

#include "TNamed.h"
#include "TRandom.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class Trigger : public virtual TNamed
{

 public:
  /** default constructor **/
  Trigger();
  /** destructor **/
  ~Trigger() override;

  /** getters **/
  Double_t getDownscale() const { return mDownscale; };
  UInt_t getNumberOfTimeSlots() const { return mNumberOfTimeSlots; };
  UInt_t getActiveTimeSlot() const { return mActiveTimeSlot; };
  Bool_t getInvertActive() const { return mInvertActive; };

  /** setters **/
  void setDownscale(Double_t val) { mDownscale = val; };
  void setNumberOfTimeSlots(UInt_t val) { mNumberOfTimeSlots = val; };
  void setActiveTimeSlot(UInt_t val) { mActiveTimeSlot = val; };
  void setInvertActive(Bool_t val) { mInvertActive = val; };

 protected:
  /** copy constructor **/
  Trigger(const Trigger&);
  /** operator= **/
  Trigger& operator=(const Trigger&);

  /** methods **/
  Bool_t isActive();
  Bool_t isDownscaled() { return gRandom->Uniform() > mDownscale; };

  /** data members **/
  Double_t mDownscale;
  UInt_t mNumberOfTimeSlots;
  UInt_t mActiveTimeSlot;
  Bool_t mInvertActive;

 private:
  UInt_t mTimeSlot;

  ClassDefOverride(Trigger, 1);

}; /** class Trigger **/

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

#endif /* ALICEO2_EVENTGEN_TRIGGER_H_ */
