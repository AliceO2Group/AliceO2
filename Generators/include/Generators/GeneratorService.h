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

/// \author S. Wenzel - September 2023

#ifndef ALICEO2_GENERATORSERVICE_H_
#define ALICEO2_GENERATORSERVICE_H_

#include <utility> // for pair
#include <vector>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCTrack.h>
#include <Generators/PrimaryGenerator.h> // could be forward declaration
#include <DetectorsBase/Stack.h>

namespace o2
{

namespace dataformats
{
class MeanVertexObject;
}

namespace eventgen
{

/// Helper class
struct VertexOption {
  virtual ~VertexOption() = default; /* this is making the class polymorphic */
};

// some specialized structs allow to pass different options
struct NoVertexOption : public VertexOption {

}; // this means to apply no Vertex

struct DiamondParamVertexOption : public VertexOption {
};

struct MeanVertexObjectOption : public VertexOption {
  // the mean vertex object
  o2::dataformats::MeanVertexObject* meanVertexObject = nullptr;
};

struct CollisionContextVertexOption : public VertexOption {
};

/// @brief A class offering convenient generator configuration and encapsulation of lower level classes.
/// Meant to reduce code duplication for places where events need to be generated.
class GeneratorService
{

 public:
  void initService(std::string const& generatorName,
                   std::string const& triggerName,
                   VertexOption const& vtxOption);

  std::pair<std::vector<MCTrack>, o2::dataformats::MCEventHeader> generateEvent();

  void generateEvent_MCTracks(std::vector<MCTrack>& tracks, o2::dataformats::MCEventHeader& header);
  void generateEvent_TParticles(std::vector<TParticle>& tparts, o2::dataformats::MCEventHeader& header);

 private:
  PrimaryGenerator mPrimGen;
  o2::data::Stack mStack;
};

} // namespace eventgen
} // namespace o2

#endif
