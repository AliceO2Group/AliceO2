// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include <functional>
#include <vector>
#include <unordered_map>

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
  has to handle the generation of input tracks (reading from input
  file) and the handing of the tracks to the FairPrimaryGenerator. It is
  called from FairMCApplication.
  *@param pStack The stack
  *@return kTRUE if successful, kFALSE if not
  **/
  Bool_t ReadEvent(FairPrimaryGenerator* primGen) final;

  /** methods to override **/
  virtual Bool_t generateEvent() = 0;   // generates event (in structure internal to generator)
  virtual Bool_t importParticles() = 0; // fills the mParticles vector (transfer from generator state)
  virtual void updateHeader(o2::dataformats::MCEventHeader* eventHeader) {};
  Bool_t triggerEvent();

  /** setters **/
  void setMomentumUnit(double val) { mMomentumUnit = val; };
  double getMomentumUnit() const { return mMomentumUnit; }
  void setEnergyUnit(double val) { mEnergyUnit = val; };
  double getEnergyUnit() const { return mEnergyUnit; }
  void setPositionUnit(double val) { mPositionUnit = val; };
  double getPositionUnit() const { return mPositionUnit; }
  void setTimeUnit(double val) { mTimeUnit = val; };
  double getTimeUnit() const { return mTimeUnit; }
  void setBoost(Double_t val) { mBoost = val; };
  void setTriggerMode(ETriggerMode_t val) { mTriggerMode = val; };
  void addTrigger(Trigger trigger) { mTriggers.push_back(trigger); };
  void addDeepTrigger(DeepTrigger trigger) { mDeepTriggers.push_back(trigger); };
  // setter for global number of events
  static void setTotalNEvents(unsigned int& n) { gTotalNEvents = n; }

  /** getters **/
  const std::vector<TParticle>& getParticles() const { return mParticles; }; //!
  static unsigned int getTotalNEvents() { return gTotalNEvents; };

  /** other **/
  void clearParticles() { mParticles.clear(); };

  /** notification methods **/
  virtual void notifyEmbedding(const o2::dataformats::MCEventHeader* eventHeader){};

  void setTriggerOkHook(std::function<void(std::vector<TParticle> const& p, int eventCount)> f) { mTriggerOkHook = f; }
  void setTriggerFalseHook(std::function<void(std::vector<TParticle> const& p, int eventCount)> f) { mTriggerFalseHook = f; }

 protected:
  /** copy constructor **/
  Generator(const Generator&);
  /** operator= **/
  Generator& operator=(const Generator&);

  /** internal methods **/
  Bool_t addTracks(FairPrimaryGenerator* primGen);
  Bool_t boostEvent();

  /** to handle cocktail constituents **/
  void addSubGenerator(int subGeneratorId, std::string const& subGeneratorDescription);
  void notifySubGenerator(int subGeneratorId) { mSubGeneratorId = subGeneratorId; }

  /** generator interface **/
  void* mInterface = nullptr;
  std::string mInterfaceName;

  /** trigger data members **/
  ETriggerMode_t mTriggerMode = kTriggerOFF;
  std::vector<Trigger> mTriggers;         //!
  std::vector<DeepTrigger> mDeepTriggers; //!

  // we allow to register callbacks so as to take specific user actions when
  // a trigger was ok nor not
  std::function<void(std::vector<TParticle> const& p, int eventCount)> mTriggerOkHook = [](std::vector<TParticle> const& p, int eventCount) {};
  std::function<void(std::vector<TParticle> const& p, int eventCount)> mTriggerFalseHook = [](std::vector<TParticle> const& p, int eventCount) {};
  int mReadEventCounter = 0; // counting the number of times

  /** conversion data members **/
  double mMomentumUnit = 1.;        // [GeV/c]
  double mEnergyUnit = 1.;          // [GeV/c]
  double mPositionUnit = 0.1;       // [cm]
  double mTimeUnit = 3.3356410e-12; // [s]

  /** particle array **/
  std::vector<TParticle> mParticles; //!

  /** lorentz boost data members **/
  Double_t mBoost;

  // a unique generator instance counter
  // this can be used to make sure no two generator instances have the same seed etc.
  static std::atomic<int> InstanceCounter;
  int mThisInstanceID = 0;

 private:
  void updateSubGeneratorInformation(o2::dataformats::MCEventHeader* header) const;

  // collect an ID and a short description of sub-generator entities
  std::unordered_map<int, std::string> mSubGeneratorsIdToDesc;
  // the current ID of the sub-generator used in the current event (if applicable)
  int mSubGeneratorId = -1;

  // global static information about (upper limit of) number of events to be generated
  static unsigned int gTotalNEvents;

  ClassDefOverride(Generator, 2);

}; /** class Generator **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATOR_H_ */
