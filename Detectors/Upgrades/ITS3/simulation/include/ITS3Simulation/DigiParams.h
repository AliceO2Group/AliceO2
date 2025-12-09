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

#ifndef ITS3_DIGIPARAMS_H
#define ITS3_DIGIPARAMS_H

#include "ITSMFTSimulation/DigiParams.h"
#include "ITS3Simulation/ChipSimResponse.h"

namespace o2::its3
{

class DigiParams final : public o2::itsmft::DigiParams
{
 private:
  float mIBNoisePerPixel = 1.e-8;
  int mIBChargeThreshold = 150;   ///< charge threshold in Nelectrons
  int mIBMinChargeToAccount = 15; ///< minimum charge contribution to account
  int mIBNSimSteps = 18;          ///< number of steps in response simulation
  float mIBNSimStepsInv = 0;      ///< its inverse

 public:
  DigiParams();

  void setIBNoisePerPixel(float v) { mIBNoisePerPixel = v; }
  float getIBNoisePerPixel() const { return mIBNoisePerPixel; }

  void setIBChargeThreshold(int v, float frac2Account = 0.1);
  int getIBChargeThreshold() const { return mIBChargeThreshold; }

  void setIBNSimSteps(int v);
  int getIBNSimSteps() const { return mIBNSimSteps; }
  float getIBNSimStepsInv() const { return mIBNSimStepsInv; }

  int getIBMinChargeToAccount() const { return mIBMinChargeToAccount; }

  const o2::itsmft::AlpideSimResponse* getAlpSimResponse() const = delete;
  void setAlpSimResponse(const o2::itsmft::AlpideSimResponse* par) = delete;

  const o2::itsmft::AlpideSimResponse* getOBSimResponse() const { return mOBSimResponse; }
  void setOBSimResponse(const o2::itsmft::AlpideSimResponse* response) { mOBSimResponse = response; }

  o2::its3::ChipSimResponse* getIBSimResponse() const { return mIBSimResponse.get(); }
  void setIBSimResponse(const o2::itsmft::AlpideSimResponse* resp);

  bool hasResponseFunctions() const { return mIBSimResponse != nullptr && mOBSimResponse != nullptr; }

  void print() const final;

 private:
  const o2::itsmft::AlpideSimResponse* mOBSimResponse = nullptr;       //!< pointer to external response
  const o2::itsmft::AlpideSimResponse* mIBSimResponseExt = nullptr;    //!< pointer to external response
  std::unique_ptr<o2::its3::ChipSimResponse> mIBSimResponse = nullptr; //!< pointer to external response

  ClassDef(DigiParams, 1);
};

} // namespace o2::its3

#endif
