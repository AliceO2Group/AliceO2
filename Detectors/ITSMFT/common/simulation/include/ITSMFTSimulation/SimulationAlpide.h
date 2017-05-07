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
    
    class SimulationAlpide : public TObject {
    public:
      enum {
        Threshold,
        ACSFromBGPar0,
        ACSFromBGPar1,
        ACSFromBGPar2,
        NumberOfParameters
      };
      SimulationAlpide();
      SimulationAlpide(Double_t param[NumberOfParameters], SegmentationPixel *, Chip *);
      SimulationAlpide(const SimulationAlpide&);
      ~SimulationAlpide() override {}

      SimulationAlpide& operator=(const SimulationAlpide&) = delete;

      void      generateClusters(DigitContainer *);
      void      clearSimulation() { mChip->Clear(); }

    private:
      Double_t  getACSFromBetaGamma(Double_t, Double_t) const; // Returns the average cluster size from the betagamma value
      Int_t     sampleCSFromLandau(Double_t, Double_t) const; // Sample the actual cluster size from a Landau distribution
      Double_t  computeIncidenceAngle(TLorentzVector) const; // Compute the angle between the particle and the normal to the chip
      Int_t     getPixelPositionResponse(Int_t, Int_t, Float_t, Float_t, Double_t) const;

    protected:
      Double_t           mParam[NumberOfParameters]; // Chip response parameters
      SegmentationPixel *mSeg;      //! Segmentation
      Chip              *mChip;     //! Chip being processed

      ClassDefOverride(SimulationAlpide,1)   // Simulation of pixel clusters
    };
  }
}
#endif
