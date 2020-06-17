// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2020

#ifndef ALICEO2_EVENTGEN_DECAYERPYTHIA8_H_
#define ALICEO2_EVENTGEN_DECAYERPYTHIA8_H_

#include "TVirtualMCDecayer.h"
#include "Pythia8/Pythia.h"

#include <iostream>

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

class DecayerPythia8 : public TVirtualMCDecayer
{

 public:
  /** default constructor **/
  DecayerPythia8() = default;
  /** destructor **/
  ~DecayerPythia8() override = default;

  /** methods to override **/
  void Init() override;
  void Decay(Int_t pdg, TLorentzVector* lv) override;
  Int_t ImportParticles(TClonesArray* particles) override;
  void SetForceDecay(Int_t type) override{};
  void ForceDecay() override{};
  Float_t GetPartialBranchingRatio(Int_t ipart) override { return 0.; };
  Float_t GetLifetime(Int_t kf) override { return 0.; };
  void ReadDecayTable() override{};

 protected:
  /** copy constructor **/
  DecayerPythia8(const DecayerPythia8&);
  /** operator= **/
  DecayerPythia8& operator=(const DecayerPythia8&);

  /** Pythia8 **/
  ::Pythia8::Pythia mPythia; //!

  ClassDefOverride(DecayerPythia8, 1);

}; /** class Decayer **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_DECAYER_H_ */
