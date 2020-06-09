// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_REDUCEDINFOTABLES_H_
#define O2_FRAMEWORK_REDUCEDINFOTABLES_H_

#include "Framework/ASoA.h"
#include "MathUtils/Utils.h"
#include <cmath>

namespace o2
{
namespace aod
{
// This is required to register SOA_TABLEs inside
// the o2::aod namespace.
//DECLARE_SOA_STORE();

namespace reducedevent {
  DECLARE_SOA_COLUMN(Tag, tag, uint64_t);
  DECLARE_SOA_COLUMN(VtxX, vtxX, float);
  DECLARE_SOA_COLUMN(VtxY, vtxY, float);
  DECLARE_SOA_COLUMN(VtxZ, vtxZ, float);
  DECLARE_SOA_COLUMN(CentVZERO, centVZERO, float);
  /*DECLARE_SOA_COLUMN_FULL(EventTag, eventTag, uint64_t, "EventTag");
  DECLARE_SOA_COLUMN_FULL(Vertex, vertex, float[3], "Vertex[3]");
  DECLARE_SOA_COLUMN_FULL(CentVZERO, centVZERO, float, "CentVZERO");*/
}  // namespace reducedevent

DECLARE_SOA_TABLE(ReducedEvents, "AOD", "REDUCEDEVENT", o2::soa::Index<>,
                  reducedevent::Tag, reducedevent::VtxX, reducedevent::VtxY, reducedevent::VtxZ, reducedevent::CentVZERO);
using ReducedEvent = ReducedEvents::iterator;


namespace reducedtrack {
  DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);
  DECLARE_SOA_COLUMN(Index, index, uint16_t);
  DECLARE_SOA_COLUMN(P1, p1, float);
  DECLARE_SOA_COLUMN(P2, p2, float);
  DECLARE_SOA_COLUMN(P3, p3, float);
  DECLARE_SOA_COLUMN(IsCartesian, isCartesian, uint8_t);
  DECLARE_SOA_COLUMN(Charge, charge, short);
  DECLARE_SOA_COLUMN(Flags, flags, uint64_t);
  DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? p1 : abs(p1)*cos(p2)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? p2 : abs(p1)*sin(p2)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? p3 : abs(p1)*sinh(p3)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? sqrt(p1*p1+p2*p2+p3*p3) : abs(p1)*cosh(p3)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? sqrt(p1*p1+p2*p2) : p1); });
  DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { 
    if(!isCartesian) return p2;
    auto phi=atan2(p2,p1); 
    if(phi>=0.0) 
      return phi;
    else 
      return phi+2.0*static_cast<float>(M_PI);
  });
  DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { 
    if(!isCartesian) return p3;
    auto mom = sqrt(p1*p1+p2*p2+p3*p3);
    auto theta = (mom>1.0e-6 ? acos(p3/mom) : 0.0);
    auto eta = tan(0.5*theta);
    if(eta>1.0e-6)
      return -1.0*log(eta);
    else 
      return 0.0;
  });
}  //namespace reducedtrack

DECLARE_SOA_TABLE(ReducedTracks, "AOD", "REDUCEDTRACK",
                  o2::soa::Index<>, reducedtrack::ReducedEventId, reducedtrack::Index,
                  reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian, reducedtrack::Charge,
                  reducedtrack::Flags, 
                  reducedtrack::Px<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Py<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Pz<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Pmom<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Pt<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Phi<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Eta<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>);

using ReducedTrack = ReducedTracks::iterator;

} // namespace aod

} // namespace o2

#endif // O2_ANALYSIS_REDUCEDINFOTABLES_H_
