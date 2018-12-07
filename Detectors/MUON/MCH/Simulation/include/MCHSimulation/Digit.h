// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization 
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SIMULATION_MCH_DIGIT_H_
#define ALICEO2_SIMULATION_MCH_DIGIT_H_

#include <CommonDataFormat/TimeStamp.h>//TODO compiler doesn't link properly 
#include "Rtypes.h"
#include <iosfwd>

namespace o2 {
  namespace mch {
    // \class Digit
    /// \brief MCH digit implementation
    using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase //: public o2::dataformats::TimeStamp<double>
    {
    public:
      Digit() = default;
      
      Digit(int pad, double adc); //check if need uint32_to
      ~Digit() = default;
      
      int GetPadID() { return mPadID; }
      void SetPadID(int pad) { mPadID=pad;}
      
      double GetADC() { return mADC; }
      void SetADC(double adc) { mADC=adc;}

      double GetTimeStamp() {return mTime;}
      void SetTimeStamp(double time){mTime =time;}
	
    private:

      int mPadID;
      double mADC;
      double mTime;
      
      ClassDefNV(Digit,1); //does not work
    };//class Digit
    
  }//namespace mch
}//namespace o2
#endif // ALICEO2_MCH_DIGIT_H_
