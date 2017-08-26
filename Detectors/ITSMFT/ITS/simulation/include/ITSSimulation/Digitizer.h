// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.h
/// \brief Definition of the ITS digitizer
#ifndef ALICEO2_ITS_DIGITIZER_H
#define ALICEO2_ITS_DIGITIZER_H

#include <vector>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTSimulation/DigiParams.h"

class TClonesArray;

namespace o2
{
  namespace ITS
  {
    class Digitizer : public TObject
    {
    public:
      Digitizer();
      ~Digitizer() override;
      Digitizer(const Digitizer&) = delete;
      Digitizer& operator=(const Digitizer&) = delete;


      void init(Bool_t build = kTRUE);

      /// Steer conversion of points to digits
      void   process(TClonesArray* points, TClonesArray* digits);

      void   setEventTime(double t);
      double getEventTime()        const  {return mEventTime;}

      void   setContinuous(bool v) {mParams.setContinuous(v);}
      bool   isContinuous()  const {return mParams.isContinuous();}
      void   fillOutputContainer(TClonesArray* digits, UInt_t maxFrame=0xffffffff);

      void   setDigiParams(const o2::ITSMFT::DigiParams& par) {mParams = par;}
      const  o2::ITSMFT::DigiParams& getDigitParams()   const {return mParams;}

      void   setCoeffToNanoSecond(double cf)                  { mCoeffToNanoSecond = cf; }
      double getCoeffToNanoSecond()                     const { return mCoeffToNanoSecond; }
      
    private:
      
      GeometryTGeo mGeometry;                    ///< ITS upgrade geometry
      std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
      o2::ITSMFT::DigiParams mParams;            ///< digitization parameters
      double mEventTime = 0;                     ///< global event time
      double mCoeffToNanoSecond = 1.0;           ///< coefficient to convert event time (Fair) to ns
      bool   mContinuous = false;                ///< flag for continuous simulation
      UInt_t mROFrameMin = 0;                    ///< lowest RO frame of current digits
      UInt_t mROFrameMax = 0;                    ///< highest RO frame of current digits

      ClassDefOverride(Digitizer, 2);
    };
  }
}

#endif /* ALICEO2_ITS_DIGITIZER_H */
