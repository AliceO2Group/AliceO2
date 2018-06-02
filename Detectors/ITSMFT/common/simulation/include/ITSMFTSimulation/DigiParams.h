// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigiParams.h
/// \brief Simulation parameters for the ALIPIDE chip

#ifndef ALICEO2_ITSMFT_DIGIPARAMS_H
#define ALICEO2_ITSMFT_DIGIPARAMS_H

#include <Rtypes.h>
#include <ITSMFTSimulation/AlpideSignalTrapezoid.h>

////////////////////////////////////////////////////////////
//                                                        //
// Simulation params for Alpide chip                      //
//                                                        //
// This is a provisionary implementation, until proper    //
// microscopic simulation and its configuration will      //
// be implemented                                         //
//                                                        //
////////////////////////////////////////////////////////////

namespace o2 {
namespace ITSMFT {

  class AlpideSimResponse;
  
  class DigiParams {
  public:

    DigiParams() = default;
    ~DigiParams() = default;
    
    void  setNoisePerPixel(float v)       {mNoisePerPixel = v;}
    float getNoisePerPixel()        const {return mNoisePerPixel;}

    void  setContinuous(bool v)           {mIsContinuous = v;}
    bool  isContinuous()            const {return mIsContinuous;}

    void setROFrameLenght(float l);
    void setROFrameDeadTime(float l) { mROFrameDeadTime = l; }
    float getROFrameLenght() const { return mROFrameLenght; }
    float getROFrameLenghtInv() const { return mROFrameLenghtInv; }
    float getROFrameDeadTime() const { return mROFrameDeadTime; }

    void   setTimeOffset(double t)        {mTimeOffset = t;}
    double getTimeOffset()          const {return mTimeOffset;}

    void setChargeThreshold(int v, float frac2Account = 0.1);
    void setNSimSteps(int v);
    void setEnergyToNElectrons(float v)   {mEnergyToNElectrons = v;}

    int   getChargeThreshold()      const {return mChargeThreshold;}
    int getMinChargeToAccount() const { return mMinChargeToAccount; }
    int   getNSimSteps()            const {return mNSimSteps;}
    float getNSimStepsInv() const { return mNSimStepsInv; }
    float getEnergyToNElectrons()   const {return mEnergyToNElectrons;}

    bool  isTimeOffsetSet()         const {return mTimeOffset>-infTime;}

    const o2::ITSMFT::AlpideSimResponse* getAlpSimResponse() const { return mAlpSimResponse; }
    void setAlpSimResponse(const o2::ITSMFT::AlpideSimResponse* par) { mAlpSimResponse=par; }

    const o2::ITSMFT::AlpideSignalTrapezoid& getSignalShape() const { return mSignalShape; }
    o2::ITSMFT::AlpideSignalTrapezoid& getSignalShape() { return (o2::ITSMFT::AlpideSignalTrapezoid&)mSignalShape; }

   private:
    static constexpr double infTime = 1e99;
    bool   mIsContinuous = false;   ///< flag for continuous simulation
    float  mNoisePerPixel = 1.e-7;  ///< ALPIDE Noise per chip
    float mROFrameLenght = 4000.;   ///< length of RO frame in ns
    float  mROFrameDeadTime = 25;   ///< dead time in end of the ROFrame, in ns
    Double_t mTimeOffset = -2*infTime;   ///< time offset to calculate ROFrame from hit time

    int mChargeThreshold = 150;  ///< charge threshold in Nelectrons
    int mMinChargeToAccount = 15; ///< minimum charge contribution to account
    int mNSimSteps       = 7;    ///< number of steps in response simulation
    float mEnergyToNElectrons = 1./3.6e-9; // conversion of eloss to Nelectrons

    o2::ITSMFT::AlpideSignalTrapezoid mSignalShape; ///< signal timeshape parameterization

    const o2::ITSMFT::AlpideSimResponse* mAlpSimResponse = nullptr; //!< pointer on external response

    // auxiliary precalculated parameters
    float mROFrameLenghtInv = 1. / 4000; ///< inverse length of RO frame in ns
    float mNSimStepsInv = 1.f / 7;       ///< its inverse

    ClassDefNV(DigiParams,1);
  };


}
}

#endif
