// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization 
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MCH_DIGIT_H_
#define ALICEO2_MCH_DIGIT_H_

//#include <CommonDataFormat/TimeStamp.h>//TODO compiler doesn't link properly 

#include <iosfwd>

namespace o2 {
  namespace mch {
    // \class Digit
    /// \brief MCH digit implementation
    class Digit //: public o2::dataformats::TimeStamp<double>
    {
    public:
      Digit() = default;

      Digit(uint32_t pad, Double_t adc, Double_t time);
      ~Digit() = default;

      Int_t GetPadID() { return mPadID; }
      void SetPadID(uint32_t pad) { mPadID=pad;}
      
      Double_t GetADC() { return mADC; }
      void SetADC(Double_t adc) { mADC=adc;}

      Double_t GetTimeStamp() {return mTime;}
      void SetTimeStamp(Double_t time){mTime =time;}
	
    private:

      Int_t mPadID;
      Double_t mADC;
      Double_t mTime;

      ClassDefNV(Digit,1);
      
    };//class Digit

    std::ostream &operator<<(std::ostream &stream, const Digit &dig);
  }//namespace mch
}//namespace o2
