// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    VisualisationEventOpenGLSerializer.h
/// \author  Julian Myrcha
///

#ifndef O2EVE_VISUALISATIONEVENTOPENGLSERIALIZER_H
#define O2EVE_VISUALISATIONEVENTOPENGLSERIALIZER_H

#include "EventVisualisationDataConverter/VisualisationEventSerializer.h"
#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include <string>

namespace o2
{
namespace event_visualisation
{

class VisualisationEventOpenGLSerializer : public VisualisationEventSerializer
{
  static void* createChunk(const char* lbl, unsigned size);
  static unsigned int* asUnsigned(void* chunk) { return (unsigned*)((char*)chunk + 8); }
  static float* asFloat(void* chunk) { return (float*)((char*)chunk + 8); }
  static unsigned char* asByte(void* chunk) { return (unsigned char*)((char*)chunk + 8); }
  static signed char* asSignedByte(void* chunk) { return (signed char*)((char*)chunk + 8); }
  static unsigned chunkSize(void* chunk);

 public:
  const std::string serializerName() const override { return std::string("VisualisationEventOpenGLSerializer"); }
  bool fromFile(VisualisationEvent& event, std::string fileName) override;
  void toFile(const VisualisationEvent& event, std::string fileName) override;
  ~VisualisationEventOpenGLSerializer() override = default;
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_VISUALISATIONEVENTOPENGLSERIALIZER_H
