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

/// \author R+Preghenella - June 2017

#ifndef ALICEO2_EVENTGEN_GENERATORTGENERATOR_H_
#define ALICEO2_EVENTGEN_GENERATORTGENERATOR_H_

#include "Generators/Generator.h"

class TGenerator;
class TClonesArray;

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class GeneratorTGenerator : public Generator
{

 public:
  /** default constructor **/
  GeneratorTGenerator();
  /** constructor with name and title **/
  GeneratorTGenerator(const Char_t* name, const Char_t* title = "ALICEo2 TGenerator Generator");
  /** destructor **/
  ~GeneratorTGenerator() override;

  /** methods to override **/
  Bool_t generateEvent() override;
  Bool_t importParticles() override;

  /** setters **/
  void setTGenerator(TGenerator* val) { mTGenerator = val; };
  const TGenerator* getTGenerator() const { return mTGenerator; }
  /** Initialize the generator if needed **/
  Bool_t Init() override;

 protected:
  /** copy constructor **/
  GeneratorTGenerator(const GeneratorTGenerator&);
  /** operator= **/
  GeneratorTGenerator& operator=(const GeneratorTGenerator&);

  /** TGenerator interface **/
  TGenerator* mTGenerator;
  TClonesArray* mCloneParticles;

  ClassDefOverride(GeneratorTGenerator, 1);

}; /** class GeneratorTGenerator **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATORTGENERATOR_H_ */
