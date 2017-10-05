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

  /** setters **/
  void setGenerator(TGenerator* val) { mGenerator = val; };

  /** Initialize the generator if needed **/
  Bool_t Init() override;

 protected:
  /** copy constructor **/
  GeneratorTGenerator(const GeneratorTGenerator&);
  /** operator= **/
  GeneratorTGenerator& operator=(const GeneratorTGenerator&);

  /** methods to override **/
  Bool_t generateEvent() override;
  Bool_t boostEvent(Double_t boost) override;
  Bool_t triggerFired(Trigger* trigger) const override;
  Bool_t addTracks(FairPrimaryGenerator* primGen) const override;

  /** methods **/
  Bool_t addHeader(PrimaryGenerator* primGen) const override;

  /** TGenerator interface **/
  TGenerator* mGenerator;
  TClonesArray* mParticles;

  ClassDefOverride(GeneratorTGenerator, 1);

}; /** class GeneratorTGenerator **/

/*****************************************************************/
/*****************************************************************/

} /** namespace eventgen **/
} /** namespace o2 **/

#endif /* ALICEO2_EVENTGEN_GENERATORTGENERATOR_H_ */
