// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#ifndef ALICEO2_EVENTGEN_GENERATOR_H_
#define ALICEO2_EVENTGEN_GENERATOR_H_

#include "FairGenerator.h"
#include <vector>
#include <array>

namespace o2
{
namespace eventgen
{
// this class implements a generic FairGenerator extension
// that provides a base class for the ALICEo2 simulation needs
// such that different interfaces to the event generators
// (i.e. TGenerator, HEPdata reader) can be implemented
// according to a common protocol
class Generator : public FairGenerator
{

 public:
  /** default constructor **/
  Generator();
  /** constructor **/
  Generator(const Char_t* name, const Char_t* title = "ALICEo2 Generator");
  /** destructor **/
  virtual ~Generator() = default;

  /** Abstract method ReadEvent must be implemented by any derived class.
	It has to handle the generation of input tracks (reading from input
	file) and the handing of the tracks to the FairPrimaryGenerator. I
	t is called from FairMCApplication.
	*@param pStack The stack
	*@return kTRUE if successful, kFALSE if not
	**/
  Bool_t ReadEvent(FairPrimaryGenerator* primGen) override;

  /** setters **/
  void setMomentumUnit(double val) { mMomentumUnit = val; };
  void setEnergyUnit(double val) { mEnergyUnit = val; };
  void setPositionUnit(double val) { mPositionUnit = val; };
  void setTimeUnit(double val) { mTimeUnit = val; };
  void setBoost(Double_t val) { mBoost = val; };

 protected:
  /** copy constructor **/
  Generator(const Generator&);
  /** operator= **/
  Generator& operator=(const Generator&);

  /** methods to override **/
  virtual Bool_t generateEvent() = 0;
  virtual Bool_t boostEvent(Double_t boost) = 0;
  virtual Bool_t addTracks(FairPrimaryGenerator* primGen) const = 0;

  /** conversion data members **/
  double mMomentumUnit = 1.;        // [GeV/c]
  double mEnergyUnit = 1.;          // [GeV/c]
  double mPositionUnit = 0.1;       // [cm]
  double mTimeUnit = 3.3356410e-12; // [s]

  /** lorentz boost data members **/
  Double_t mBoost;

  ClassDefOverride(Generator, 1);

}; /** class Generator **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATOR_H_ */
