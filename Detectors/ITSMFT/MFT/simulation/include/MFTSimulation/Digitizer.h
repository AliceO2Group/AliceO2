/// \file Digitizer.h
/// \brief Implementation of the convertion from Point to Digit
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITIZER_H_
#define ALICEO2_MFT_DIGITIZER_H_

class TClonesArray;

namespace o2 {
namespace MFT {

class Digitizer : public TObject
{

public:

  Digitizer();
  ~Digitizer() override;
  
  void init(Bool_t build = kTRUE);
  
  /// Steer conversion of points to digits
  /// @param points Container with ITS points
  /// @return digits container
  DigitContainer& process(TClonesArray* points);
  void process(TClonesArray* points, TClonesArray* digits);
  
private:

  Digitizer(const Digitizer&);
  Digitizer& operator=(const Digitizer&);
  
  GeometryTGeo mGeometry;                     ///< ITS upgrade geometry
  Int_t mNumOfChips;                          ///< Number of chips
  std::vector<o2::ITSMFT::Chip> mChips;  ///< Array of chips
  std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
  DigitContainer mDigitContainer;             ///< Internal digit storage

  ClassDefOverride(Digitizer, 1)

};

}
}
