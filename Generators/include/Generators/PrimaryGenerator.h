// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2017

#ifndef ALICEO2_EVENTGEN_PRIMARYGENERATOR_H_
#define ALICEO2_EVENTGEN_PRIMARYGENERATOR_H_

#include "FairPrimaryGenerator.h"

namespace o2
{
namespace eventgen
{

/** 
 ** custom primary generator in order to be able to deal with
 ** specific O2 matters, like initialisation, generation, ...
 **/

class PrimaryGenerator : public FairPrimaryGenerator
{

 public:
  /** default constructor **/
  PrimaryGenerator() = default;
  /** destructor **/
  virtual ~PrimaryGenerator() = default;

  /** initialize the generator **/
  virtual Bool_t Init() override;

 protected:
  /** copy constructor **/
  PrimaryGenerator(const PrimaryGenerator&) = default;
  /** operator= **/
  PrimaryGenerator& operator=(const PrimaryGenerator&) = default;

  /** set interaction diamond position **/
  void setInteractionDiamond(const Double_t* xyz, const Double_t* sigmaxyz);

  ClassDefOverride(PrimaryGenerator, 1);

}; /** class PrimaryGenerator **/

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

#endif /* ALICEO2_EVENTGEN_PRIMARYGENERATOR_H_ */
