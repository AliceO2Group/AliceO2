// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DIGIT_H_
#define ALICEO2_EMCAL_DIGIT_H_

#include "FairTimeStamp.h"
#include "Rtypes.h"
#include <iosfwd>

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif

namespace o2 {
  namespace EMCAL {
    /// \class Digit
    /// \brief EMCAL digit implementation
    class Digit : public FairTimeStamp {
    public:
      Digit() = default;

      Digit(Int_t tower, Double_t amplitude, Double_t time);
      ~Digit() override = default;

      bool operator<(const Digit &ref) const;
      bool operator>(const Digit &ref) const;

      bool CanAdd(const Digit other);
      Digit& operator+=(const Digit& other); // Adds energy of other digits to this digit.
      friend Digit operator+(Digit lhs, const Digit& rhs)// Adds energy of two digits.
      {
        lhs += rhs;
	return lhs;
      }

      Int_t GetTower() const { return mTower; }
      void SetTower(Int_t tower) { mTower = tower; }

      Double_t GetAmplitude() const { return mAmplitude; }
      void SetAmplitude(Double_t amplitude) { mAmplitude = amplitude; }

      void PrintStream(std::ostream &stream) const;

    private:
#ifndef __CINT__
      friend class boost::serialization::access;
#endif

      Int_t             mTower;                 ///< Tower index (absolute cell ID)
      Double_t          mAmplitude;             ///< Amplitude

      ClassDefOverride(Digit, 1);
    };

    std::ostream &operator<<(std::ostream &stream, const Digit &dig);
  }
}
#endif
