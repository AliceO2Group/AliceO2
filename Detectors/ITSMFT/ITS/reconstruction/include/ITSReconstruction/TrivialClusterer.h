/// \file TrivialClusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_TRIVIALCLUSTERER_H
#define ALICEO2_ITS_TRIVIALCLUSTERER_H

#include "Rtypes.h"  // for TrivialClusterer::Class, Double_t, ClassDef, etc

class TClonesArray;

namespace AliceO2 {
  namespace ITSMFT {
    class SegmentationPixel;
  }
}
using AliceO2::ITSMFT::SegmentationPixel;

namespace AliceO2
{
namespace ITS
{
  class TrivialClusterer
{
 public:
  TrivialClusterer();
  ~TrivialClusterer();

  TrivialClusterer(const TrivialClusterer&) = delete;
  TrivialClusterer& operator=(const TrivialClusterer&) = delete;

  /// Steer conversion of points to digits
  /// @param points Container with ITS points
  /// @return digits container
  void process(const SegmentationPixel *seg, const TClonesArray* digits, TClonesArray* clusters);

};
}
}

#endif /* ALICEO2_ITS_TRIVIALCLUSTERER_H */
