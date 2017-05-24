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
    class DigitContainer;

    class SimulationAlpide : public Chip {
    public:
      enum {
        Threshold,
        ACSFromBGPar0,
        ACSFromBGPar1,
        ACSFromBGPar2,
        NumberOfParameters
      };
      SimulationAlpide();
      SimulationAlpide(Double_t param[NumberOfParameters], Int_t index, const TGeoHMatrix *m);
      SimulationAlpide(const SimulationAlpide&);
      ~SimulationAlpide() override {}

      SimulationAlpide& operator=(const SimulationAlpide&) = delete;

      void      generateClusters(const SegmentationPixel *, DigitContainer *);
      void      clearSimulation() { Chip::Clear(); }

    private:
      void      addNoise(Double_t, const SegmentationPixel*, DigitContainer*); // Add noise to the chip
      Double_t  betaGammaFunction(Double_t, Double_t, Double_t, Double_t) const;
      Double_t  gaussian2D(Double_t, Double_t, Double_t, Double_t) const;
      Double_t  getACSFromBetaGamma(Double_t, Double_t) const; // Returns the average cluster size from the betagamma value
      Int_t     sampleCSFromLandau(Double_t, Double_t) const; // Sample the actual cluster size from a Landau distribution
      Double_t  computeIncidenceAngle(TLorentzVector) const; // Compute the angle between the particle and the normal to the chip
      Int_t     getPixelPositionResponse(const SegmentationPixel *, Int_t, Int_t, Float_t, Float_t, Double_t) const;

    protected:
      Double_t  mParam[NumberOfParameters]; // Chip response parameters
    };
  }
}
#endif
