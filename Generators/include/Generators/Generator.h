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
#include "TParticle.h"
#include "Generators/Trigger.h"
#include <vector>

namespace o2
{
namespace dataformats
{
class MCEventHeader;
}
} // namespace o2

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

// this class implements a generic FairGenerator extension
// that provides a base class for the ALICEo2 simulation needs
// such that different interfaces to the event generators
// (i.e. TGenerator, HEPdata reader) can be implemented
// according to a common protocol

class Generator : public FairGenerator
{

 public:
  enum ETriggerMode_t { kTriggerOFF,
                        kTriggerOR,
                        kTriggerAND };

  /** default constructor **/
  Generator();
  /** constructor **/
  Generator(const Char_t* name, const Char_t* title = "ALICEo2 Generator");
  /** destructor **/
  ~Generator() override = default;

  /** Initialize the generator if needed **/
  Bool_t Init() override;

  /** Abstract method ReadEvent must be implemented by any derived class.
	It has to handle the generation of input tracks (reading from input
	file) and the handing of the tracks to the FairPrimaryGenerator. I
	t is called from FairMCApplication.
	*@param pStack The stack
	*@return kTRUE if successful, kFALSE if not
	**/
  Bool_t ReadEvent(FairPrimaryGenerator* primGen) final;

  /** methods to override **/
  virtual Bool_t generateEvent() = 0;
  virtual Bool_t importParticles() = 0;

  /** setters **/
  void setMomentumUnit(double val) { mMomentumUnit = val; };
  void setEnergyUnit(double val) { mEnergyUnit = val; };
  void setPositionUnit(double val) { mPositionUnit = val; };
  void setTimeUnit(double val) { mTimeUnit = val; };
  void setBoost(Double_t val) { mBoost = val; };
  void setTriggerMode(ETriggerMode_t val) { mTriggerMode = val; };
  void addTrigger(Trigger trigger) { mTriggers.push_back(trigger); };
  void addDeepTrigger(DeepTrigger trigger) { mDeepTriggers.push_back(trigger); };

  /** getters **/
  const std::vector<TParticle>& getParticles() const { return mParticles; }; //!

  /** other **/
  void clearParticles() { mParticles.clear(); };

  /** notification methods **/
  virtual void notifyEmbedding(const o2::dataformats::MCEventHeader* eventHeader){};

 protected:
  /** copy constructor **/
  Generator(const Generator&);
  /** operator= **/
  Generator& operator=(const Generator&);

  /** methods that can be overridded **/
  virtual void updateHeader(o2::dataformats::MCEventHeader* eventHeader){};

  /** internal methods **/
  Bool_t addTracks(FairPrimaryGenerator* primGen);
  Bool_t boostEvent();
  Bool_t triggerEvent();

  /** generator interface **/
  void* mInterface = nullptr;
  std::string mInterfaceName;

  /** trigger data members **/
  ETriggerMode_t mTriggerMode = kTriggerOFF;
  std::vector<Trigger> mTriggers;         //!
  std::vector<DeepTrigger> mDeepTriggers; //!

  /** conversion data members **/
  double mMomentumUnit = 1.;        // [GeV/c]
  double mEnergyUnit = 1.;          // [GeV/c]
  double mPositionUnit = 0.1;       // [cm]
  double mTimeUnit = 3.3356410e-12; // [s]

  /** particle array **/
  std::vector<TParticle> mParticles; //!

  /** lorentz boost data members **/
  Double_t mBoost;

  ClassDefOverride(Generator, 1);

}; /** class Generator **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATOR_H_ */
