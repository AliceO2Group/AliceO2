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

////////////////////////////////////////////////////////////
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

    enum Hit2DigitsMethod {p2dSimple, p2dCShape};

    DigiParams() = default;
    ~DigiParams() = default;
    
    void  setNoisePerPixel(float v)       {mNoisePerPixel = v;}
    float getNoisePerPixel()        const {return mNoisePerPixel;}

    void  setContinuous(bool v)           {mIsContinuous = v;}
    bool  isContinuous()            const {return mIsContinuous;}

    void   setROFrameLenght(float l)      {mROFrameLenght = l;}
    void   setROFrameDeadTime(float l)    {mROFrameDeadTime = l;}
    float  getROFrameLenght()       const {return mROFrameLenght;}
    float  getROFrameDeadTime()     const {return mROFrameDeadTime;}

    void   setTimeOffset(double t)        {mTimeOffset = t;}
    double getTimeOffset()          const {return mTimeOffset;}

    void setACSFromBGPar0(float v)        {mACSFromBGPar0 = v;}
    void setACSFromBGPar1(float v)        {mACSFromBGPar1 = v;}
    void setACSFromBGPar2(float v)        {mACSFromBGPar2 = v;}
    void setChargeThreshold(int v)        {mChargeThreshold = v;}
    void setNSimSteps(int v)              {mNSimSteps = v;}
    void setEnergyToNElectrons(float v)   {mEnergyToNElectrons = v;}

    float getACSFromBGPar0()        const {return mACSFromBGPar0;}
    float getACSFromBGPar1()        const {return mACSFromBGPar1;}
    float getACSFromBGPar2()        const {return mACSFromBGPar2;}
    int   getChargeThreshold()      const {return mChargeThreshold;}
    int   getNSimSteps()            const {return mNSimSteps;}
    float getEnergyToNElectrons()   const {return mEnergyToNElectrons;}

    bool  isTimeOffsetSet()         const {return mTimeOffset>-infTime;}
    Hit2DigitsMethod getHit2DigitsMethod() const {return mHit2DigitsMethod;}
    void setHitDigitsMethod(Hit2DigitsMethod m) { mHit2DigitsMethod = m; }

    const o2::ITSMFT::AlpideSimResponse* getAlpSimResponse() const { return mAlpSimResponse; }
    void setAlpSimResponse(const o2::ITSMFT::AlpideSimResponse* par) { mAlpSimResponse=par; }
    
  private:
    static constexpr double infTime = 1e99;
    Hit2DigitsMethod mHit2DigitsMethod = p2dCShape; ///< method of point to digitis conversion
    bool   mIsContinuous = false;   ///< flag for continuous simulation
    float  mNoisePerPixel = 1.e-7;  ///< ALPIDE Noise per chip
    float  mROFrameLenght = 10000;  ///< length of RO frame in ns
    float  mROFrameDeadTime = 25;   ///< dead time in end of the ROFrame, in ns
    Double_t mTimeOffset = -2*infTime;   ///< time offset to calculate ROFrame from hit time

    float mACSFromBGPar0 = -1.315;
    float mACSFromBGPar1 = 0.5018;
    float mACSFromBGPar2 = 1.084;

    int mChargeThreshold = 150;  ///< charge threshold in Nelectrons
    int mNSimSteps       = 7;    ///< number of steps in response simulation
    float mEnergyToNElectrons = 1./3.6e-9; // conversion of eloss to Nelectrons

    const o2::ITSMFT::AlpideSimResponse* mAlpSimResponse = nullptr;
    
    ClassDefNV(DigiParams,1);
  };


}
}

#endif
