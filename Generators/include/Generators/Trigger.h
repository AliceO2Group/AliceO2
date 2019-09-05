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

#ifndef ALICEO2_EVENTGEN_TRIGGER_H_
#define ALICEO2_EVENTGEN_TRIGGER_H_

#include "TNamed.h"

class TClonesArray;

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class Trigger : public TNamed
{

 public:
  /** default constructor **/
  Trigger() = default;
  /** destructor **/
  ~Trigger() override = default;

  /** methods to override **/
  virtual Bool_t fired(TClonesArray* particles) = 0;

 protected:
  /** copy constructor **/
  Trigger(const Trigger&);
  /** operator= **/
  Trigger& operator=(const Trigger&);

  ClassDefOverride(Trigger, 1);

}; /** class Trigger **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_TRIGGER_H_ */
