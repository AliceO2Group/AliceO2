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

#ifndef ALICEO2_EVENTGEN_GENERATORHEPMC_H_
#define ALICEO2_EVENTGEN_GENERATORHEPMC_H_

#include "Generators/Generator.h"
#include "Generators/GeneratorFileOrCmd.h"
#include "Generators/GeneratorHepMCParam.h"

#ifdef GENERATORS_WITH_HEPMC3_DEPRECATED
namespace HepMC
{
class Reader;
class GenEvent;
class FourVector;
} // namespace HepMC
#else
namespace HepMC3
{
class Reader;
class GenEvent;
class FourVector;
class GenParticle;
} // namespace HepMC3
#endif

namespace o2
{
namespace conf
{
class SimConfig;
}
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class GeneratorHepMC : public Generator, public GeneratorFileOrCmd
{

 public:
  /** default constructor **/
  GeneratorHepMC();
  /** constructor **/
  GeneratorHepMC(const Char_t* name,
                 const Char_t* title = "ALICEo2 HepMC Generator");
  /** destructor **/
  ~GeneratorHepMC() override;

  /** Initialize the generator. **/
  Bool_t Init() override;

  /**
   * Configure the generator from parameters and the general
   * simulation configuration.  This is implemented as a member
   * function so as to better facilitate changes. */
  void setup(const GeneratorFileOrCmdParam& param0,
             const GeneratorHepMCParam& param,
             const conf::SimConfig& config);
  /**
   * Generate a single event.  The event is read in from the current
   * input file.  Returns false if a new event could not be read.
   **/
  Bool_t generateEvent() override;
  /**
   * Import particles from the last read event into a vector
   * TParticle.  Returns false if no particles could be exported to
   * the vector.
   */
  Bool_t importParticles() override;

  /** setters **/
  void setEventsToSkip(uint64_t val) { mEventsToSkip = val; };

 protected:
  /** copy constructor **/
  GeneratorHepMC(const GeneratorHepMC&);
  /** operator= **/
  GeneratorHepMC& operator=(const GeneratorHepMC&);

  /** methods **/
#ifdef GENERATORS_WITH_HEPMC3_DEPRECATED
  const HepMC::FourVector getBoostedVector(const HepMC::FourVector& vector, Double_t boost);
#else
  const HepMC3::FourVector getBoostedVector(const HepMC3::FourVector& vector, Double_t boost);
#endif

  /** methods that can be overridded **/
  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override;
  /** Make our reader */
  bool makeReader();

  /** Type of function to select particles to keep when pruning
   * events */
  typedef bool (*Select)(std::shared_ptr<const HepMC3::GenParticle>);
  /** Prune event of particles that are not selected by passed
   * function.  The event structure is preserved. */
  void pruneEvent(Select select);

  /** HepMC interface **/
  uint64_t mEventsToSkip = 0;
  /** HepMC event record version to expected.  Deprecated. */
  int mVersion = 0;
  std::shared_ptr<HepMC3::Reader> mReader;
  /** Event structure */
  HepMC3::GenEvent* mEvent = nullptr;
  /** Option whether to prune event */
  bool mPrune; //!

  ClassDefOverride(GeneratorHepMC, 1);

}; /** class GeneratorHepMC **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATORHEPMC_H_ */
