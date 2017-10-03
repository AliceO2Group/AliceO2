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

#ifndef ALICEO2_EVENTGEN_GENERATOR_H_
#define ALICEO2_EVENTGEN_GENERATOR_H_

#include "FairGenerator.h"
#include <vector>

namespace o2
{
namespace dataformats
{
class GeneratorHeader;
}
}

using o2::dataformats::GeneratorHeader;

namespace o2
{
namespace eventgen
{

class PrimaryGenerator;
class Trigger;

/*****************************************************************/
/*****************************************************************/

class Generator : public FairGenerator
{

 public:
  enum ETriggerMode_t { kTriggerOFF, kTriggerOR, kTriggerAND };

  /** default constructor **/
  Generator();
  /** constructor **/
  Generator(const Char_t* name, const Char_t* title = "ALICEo2 Generator");
  /** destructor **/
  ~Generator() override;

  /** Abstract method ReadEvent must be implemented by any derived class.
It has to handle the generation of input tracks (reading from input
file) and the handing of the tracks to the FairPrimaryGenerator. I
t is called from FairMCApplication.
*@param pStack The stack
*@return kTRUE if successful, kFALSE if not
**/
  Bool_t ReadEvent(FairPrimaryGenerator* primGen) override;

  /** getters **/
  GeneratorHeader* getHeader() const { return mHeader; };

  /** setters **/
  void setTriggerMode(ETriggerMode_t val) { mTriggerMode = val; };
  void setMaxTriggerAttempts(Int_t val) { mMaxTriggerAttempts = val; };
  void addTrigger(Trigger* trigger) { mTriggers.push_back(trigger); };
  void setBoost(Double_t val) { mBoost = val; };

 protected:
  /** copy constructor **/
  Generator(const Generator&);
  /** operator= **/
  Generator& operator=(const Generator&);

  /** methods to override **/
  virtual Bool_t generateEvent() = 0;
  virtual Bool_t boostEvent(Double_t boost) = 0;
  virtual Bool_t triggerFired(Trigger* trigger) const = 0;
  virtual Bool_t addTracks(FairPrimaryGenerator* primGen) const = 0;

  /** methods **/
  virtual Bool_t addHeader(PrimaryGenerator* primGen) const;
  Bool_t triggerEvent() const;

  /** trigger data members **/
  ETriggerMode_t mTriggerMode;
  Int_t mMaxTriggerAttempts;
  std::vector<Trigger*> mTriggers;

  /** lorentz boost data members **/
  Double_t mBoost;

  /** header data members **/
  GeneratorHeader* mHeader;

  ClassDefOverride(Generator, 1);

}; /** class Generator **/

/*****************************************************************/
/*****************************************************************/

} /** namespace eventgen **/
} /** namespace o2 **/

#endif /* ALICEO2_EVENTGEN_GENERATOR_H_ */
