// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
  
  class DigiParams { 
  public:

    enum Point2DigitsMethod {p2dSimple, p2dCShape};
    
    DigiParams() = default;
    ~DigiParams() = default;

    void  setThreshold(float v)           {mThreshold = v;}
    float getThreshold()            const {return mThreshold;}

    void  setNoisePerPixel(float v)       {mNoisePerPixel = v;}
    float getNoisePerPixel()        const {return mNoisePerPixel;}

    void  setContinuous(bool v)           {mIsContinuous = v;}
    bool  isContinuous()            const {return mIsContinuous;}

    void   setROFrameLenght(UInt_t l)     {mROFrameLenght = l;}
    void   setROFrameDeadTime(UInt_t l)   {mROFrameDeadTime = l;}    
    UInt_t getROFrameLenght()       const {return mROFrameLenght;}
    UInt_t getROFrameDeadTime()     const {return mROFrameDeadTime;}

    void   setTimeOffset(double t)        {mTimeOffset = t;}
    double getTimeOffset()          const {return mTimeOffset;}

    void setACSFromBGPar0(float v)        {mACSFromBGPar0 = v;}
    void setACSFromBGPar1(float v)        {mACSFromBGPar1 = v;}
    void setACSFromBGPar2(float v)        {mACSFromBGPar2 = v;}

    float getACSFromBGPar0()        const {return mACSFromBGPar0;}
    float getACSFromBGPar1()        const {return mACSFromBGPar1;}
    float getACSFromBGPar2()        const {return mACSFromBGPar2;}

    bool  isTimeOffsetSet()         const {return mTimeOffset>-infTime;}
    Point2DigitsMethod getPoint2DigitsMethod() const {return mPoint2DigitsMethod;}
    void setPointDigitsMethod(Point2DigitsMethod m) { mPoint2DigitsMethod = m; }
    
  private:
    static constexpr double infTime = 1e99;
    Point2DigitsMethod mPoint2DigitsMethod = p2dCShape; ///< method of point to digitis conversion
    bool   mIsContinuous = false;   ///< flag for continuous simulation
    float  mThreshold = 1.e-6;      ///< threshold in N electrons RS: TODO: Fix conversion from eloss to nelectrons 
    float  mNoisePerPixel = 1.e-7;  ///< ALPIDE Noise per chip
    float  mROFrameLenght = 10000;  ///< length of RO frame in ns
    float  mROFrameDeadTime = 25;   ///< dead time in end of the ROFrame, in ns
    Double_t mTimeOffset = -2*infTime;   ///< time offset to calculate ROFrame from hit time

    float mACSFromBGPar0 = -1.315;
    float mACSFromBGPar1 = 0.5018;
    float mACSFromBGPar2 = 1.084;
    
    ClassDefNV(DigiParams,1);
  };

  
}
}

#endif
