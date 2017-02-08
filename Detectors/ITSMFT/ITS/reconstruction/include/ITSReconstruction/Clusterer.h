/// \file Clusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_CLUSTERER_H
#define ALICEO2_ITS_CLUSTERER_H

#include "Rtypes.h"  // for Clusterer::Class, Double_t, ClassDef, etc

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
  class Clusterer
{
 public:
  Clusterer();
  ~Clusterer();

  /// Steer conversion of points to digits
  /// @param points Container with ITS points
  /// @return digits container
  void process(const SegmentationPixel *seg, const TClonesArray* digits, TClonesArray* clusters);

 private:
  Clusterer(const Clusterer&);
  Clusterer& operator=(const Clusterer&);
};
}
}

#endif /* ALICEO2_ITS_CLUSTERER_H */
