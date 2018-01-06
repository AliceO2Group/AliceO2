// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DIGITIZER_H
#define ALICEO2_EMCAL_DIGITIZER_H

#include <vector>
#include <memory>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "EMCALBase/Digit.h"

namespace o2
{
  namespace EMCAL
  {
    class Digitizer : public TObject
    {
    public:

      Digitizer() = default;
      ~Digitizer() override = default;
      Digitizer(const Digitizer&) = delete;
      Digitizer& operator=(const Digitizer&) = delete;

      void init();

      /// Steer conversion of hits to digits
      void   process(const std::vector<Hit>* hits, std::vector<Digit>* digits);

      void   setEventTime(double t);
      double getEventTime() const { return mEventTime; }

      void   setContinuous(bool v) { mContinuous = v; }
      bool   isContinuous() const { return mContinuous; }
      void   fillOutputContainer(std::vector<Digit>* digits, UInt_t maxFrame=0xffffffff);

      void   setCoeffToNanoSecond(double cf) { mCoeffToNanoSecond = cf; }
      double getCoeffToNanoSecond() const { return mCoeffToNanoSecond; }

      void setCurrSrcID(int v);
      int getCurrSrcID() const { return mCurrSrcID; }

      void setCurrEvID(int v);
      int getCurrEvID()  const { return mCurrEvID; }

      void setGeometry(const o2::EMCAL::Geometry* gm) { mGeometry = gm; }

      o2::EMCAL::Digit HitToDigit(Hit hit);
      
    private:

      const o2::EMCAL::Geometry* mGeometry = nullptr;    // EMCAL geometry
      //std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
      double mEventTime = 0;                     ///< global event time
      double mCoeffToNanoSecond = 1.0;           ///< coefficient to convert event time (Fair) to ns
      bool   mContinuous = false;                ///< flag for continuous simulation
      UInt_t mROFrameMin = 0;                    ///< lowest RO frame of current digits
      UInt_t mROFrameMax = 0;                    ///< highest RO frame of current digits
      int    mCurrSrcID = 0;                     ///< current MC source from the manager
      int    mCurrEvID = 0;                      ///< current event ID from the manager

      //std::unique_ptr<o2::ITSMFT::AlpideSimResponse> mAlpSimResp; // simulated response 
      
      ClassDefOverride(Digitizer, 1);
    };
  }
}

#endif /* ALICEO2_EMCAL_DIGITIZER_H */
