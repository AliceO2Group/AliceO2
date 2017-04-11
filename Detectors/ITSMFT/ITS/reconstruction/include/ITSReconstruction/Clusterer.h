/// \file Clusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_CLUSTERER_H
#define ALICEO2_ITS_CLUSTERER_H

#include "Rtypes.h"  // for Clusterer::Class, Double_t, ClassDef, etc

class TClonesArray;

namespace o2 {
  namespace ITSMFT {
    class SegmentationPixel;
  }
}

namespace o2
{
namespace ITS
{
  class Clusterer
{
 public:
  Clusterer();
  ~Clusterer();

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  /// Steer conversion of points to digits
  /// @param points Container with ITS points
  /// @return digits container
  void process(const o2::ITSMFT::SegmentationPixel *seg, const TClonesArray* digits, TClonesArray* clusters);

};
}
}

#endif /* ALICEO2_ITS_CLUSTERER_H */
