/// \file SimulationAlpide.h
/// \brief Simulation of the ALIPIDE chip response

#ifndef ALICEO2_ITS_ALPIDE_H
#define ALICEO2_ITS_ALPIDE_H

////////////////////////////////////////////////////////////
// Simulation class for Alpide upgrade pixels (2016)      //
//                                                        //
// Author: D. Pagano                                      //
// Contact: davide.pagano@cern.ch                         //
////////////////////////////////////////////////////////////

#include <TObject.h>

#include "ITSBase/SensMap.h"
#include "ITSSimulation/Chip.h"

class TLorentzVector;
class TClonesArray;
class TSeqCollection;


namespace AliceO2 {
  namespace ITS {

    class SegmentationPixel;

    //-------------------------------------------------------------------

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
      SimulationAlpide
      (Double_t param[NumberOfParameters], SegmentationPixel *, Chip *);
      SimulationAlpide(const SimulationAlpide&);
      virtual   ~SimulationAlpide();

      void Init(Double_t param[NumberOfParameters], SegmentationPixel *, Chip *);

      void      SDigitiseChip(TClonesArray*);
      void      FinishSDigitiseChip(TClonesArray*);
      void      DigitiseChip(TClonesArray*);
      Bool_t    AddSDigitsToChip(TSeqCollection*, Int_t);
      void      GenerateCluster();
      void      clearSimulation() { fSensMap->clear(); fChip->Clear(); }

    private:
      void      FrompListToDigits(TClonesArray*);
      void      WriteSDigits(TClonesArray*);
      Double_t  ACSFromBetaGamma(Double_t, Double_t) const; // Returns the average cluster size from the betagamma value
      Int_t     CSSampleFromLandau(Double_t, Double_t) const; // Sample the actual cluster size from a Landau distribution
      Double_t  ComputeIncidenceAngle(TLorentzVector) const; // Compute the angle between the particle and the normal to the chip
      Int_t     GetPixelPositionResponse(Int_t, Int_t, Float_t, Float_t, Double_t) const;
      void      CreateDigi(UInt_t, UInt_t, Int_t, Int_t);

    protected:
      Double_t           fParam[NumberOfParameters]; // Chip response parameters
      SegmentationPixel *fSeg;      //! Segmentation
      SensMap           *fSensMap;  //! Sensor map for hits manipulations
      Chip              *fChip;     //! Chip being processed

      ClassDef(SimulationAlpide,1)   // Simulation of pixel clusters
    };
  }
}
#endif
