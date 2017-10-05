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

#ifndef ALICEO2_EVENTGEN_TRIGGERTGENERATOR_H_
#define ALICEO2_EVENTGEN_TRIGGERTGENERATOR_H_

#include "Generators/Trigger.h"

class TClonesArray;
class TGenerator;

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class TriggerTGenerator : public virtual Trigger
{

 public:
  /** default constructor **/
  TriggerTGenerator();
  /** destructor **/
  ~TriggerTGenerator() override;

  /** methods **/
  Bool_t triggerEvent(TClonesArray* particles, TGenerator* generator);

 protected:
  /** copy constructor **/
  TriggerTGenerator(const TriggerTGenerator&);
  /** operator= **/
  TriggerTGenerator& operator=(const TriggerTGenerator&);

  /** methods **/
  virtual Bool_t isTriggered(TClonesArray* particles, TGenerator* generator) const = 0;

 private:
  ClassDefOverride(TriggerTGenerator, 1);

}; /** class TriggerTGenerator **/

/*****************************************************************/
/*****************************************************************/

} /** namespace eventgen **/
} /** namespace o2 **/

#endif /* ALICEO2_EVENTGEN_TRIGGERTGENERATOR_H_ */
