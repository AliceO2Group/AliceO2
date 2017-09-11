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

class TLorentzVector;
class TClonesArray;
class TSeqCollection;

namespace o2 {
  namespace ITSMFT {

    //-------------------------------------------------------------------

    class SegmentationPixel;
    
    class SimulationAlpide : public Chip {
    public:
      SimulationAlpide() = default;
      SimulationAlpide(const DigiParams* par, Int_t index, const TGeoHMatrix *m)
	: Chip(par, index, m) {}
	
      ~SimulationAlpide() = default;

      SimulationAlpide& operator=(const SimulationAlpide&) = delete;

      void      Hits2Digits(const SegmentationPixel *seg, double eventTime, UInt_t &minFr, UInt_t &maxFr);

      void      addNoise(const SegmentationPixel* seg, UInt_t rofMin, UInt_t rofMax);
      
      void      clearSimulation() { Chip::Clear(); }
      
    private:
      
      void      Hit2DigitsCShape(const Hit *hit, UInt_t roFrame, double eventTime, const SegmentationPixel* seg);
      void      Hit2DigitsSimple(const Hit *hit, UInt_t roFrame, double eventTime, const SegmentationPixel* seg);

      
      Double_t  betaGammaFunction(Double_t, Double_t, Double_t, Double_t) const;
      Double_t  gaussian2D(Double_t, Double_t, Double_t, Double_t) const;
      Double_t  getACSFromBetaGamma(Double_t) const; // Returns the average cluster size from the betagamma value
      void      updateACSWithAngle(Double_t&, Double_t) const; // Modify the ACS according to the effective incidence angles
      Int_t     sampleCSFromLandau(Double_t, Double_t) const; // Sample the actual cluster size from a Landau distribution
      Double_t  computeIncidenceAngle(TLorentzVector) const; // Compute the angle between the particle and the normal to the chip
      Int_t     getPixelPositionResponse(const SegmentationPixel *, Int_t, Int_t, Float_t, Float_t, Double_t) const;      
    };
  }
}
#endif
