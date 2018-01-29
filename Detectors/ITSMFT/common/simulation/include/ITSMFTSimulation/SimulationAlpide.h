// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimulationAlpide.h
/// \brief Simulation of the ALIPIDE chip response

#ifndef ALICEO2_ITSMFT_ALPIDE_H
#define ALICEO2_ITSMFT_ALPIDE_H

////////////////////////////////////////////////////////////
// Simulation class for Alpide upgrade pixels (2016)      //
//                                                        //
// Author: D. Pagano                                      //
// Contact: davide.pagano@cern.ch                         //
////////////////////////////////////////////////////////////

#include <TObject.h>

#include "ITSMFTSimulation/Chip.h"
#include "ITSMFTSimulation/AlpideSimResponse.h"

class TLorentzVector;
class TSeqCollection;

namespace o2 {
  namespace ITSMFT {

    //-------------------------------------------------------------------

    class SimulationAlpide : public Chip {
    public:
      SimulationAlpide() = default;
      SimulationAlpide(const DigiParams* par, Int_t index, const o2::Transform3D *m)
	: Chip(par, index, m) {}

      ~SimulationAlpide() = default;

      SimulationAlpide& operator=(const SimulationAlpide&) = delete;

      void      Hits2Digits(double eventTime, UInt_t &minFr, UInt_t &maxFr);

      void      addNoise(UInt_t rofMin, UInt_t rofMax);

      void      clearSimulation() { Chip::Clear(); }

    private:

      void      Hit2DigitsCShape(const Hit *hit, UInt_t roFrame, double eventTime);
      void      Hit2DigitsSimple(const Hit *hit, UInt_t roFrame, double eventTime);


      Double_t  computeIncidenceAngle(TLorentzVector) const; // Compute the angle between the particle and the normal to the chip
    };
  }
}
#endif
