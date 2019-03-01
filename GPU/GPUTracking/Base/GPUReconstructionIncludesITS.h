#ifndef GPURECONSTRUCTIONINCLDUESITS_H
#define GPURECONSTRUCTIONINCLDUESITS_H

#ifdef HAVE_O2HEADERS
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/VertexerTraits.h"
#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsCPU : public TrackerTraits {}; class VertexerTraits {}; }}
#endif

#endif
