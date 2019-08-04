// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGERBUILDER_H
#define ALICEO2_MERGERBUILDER_H

/// \file MergeBuilder.h
/// \brief Definition of O2 Mergers builder, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerConfig.h"

#include <Framework/DataProcessorSpec.h>
#include <Framework/CompletionPolicy.h>

#include <string>

namespace o2
{
namespace experimental::mergers
{

/// \brief A builder class to generate a DataProcessorSpec of one Merger
///
/// A builder class to generate a DataProcessorSpec of one Merger. One builder can be reused by using setters to change
/// the configuration and generating more Mergers. If OutputSpec is not set or it is has either header::gDataOriginInvalid
/// or header::gDataDescriptionInvalid, OutputSpec is generated using.
class MergerBuilder
{
 public:
  /// \brief Default constructor.
  MergerBuilder();
  /// \brief Default destructor.
  ~MergerBuilder() = default;

  void setName(std::string);
  void setInputSpecs(const framework::Inputs&);
  void setOutputSpec(const framework::OutputSpec&);
  void setTopologyPosition(size_t layer, size_t id);
  void setConfig(MergerConfig);

  framework::DataProcessorSpec buildSpec();

  /// \brief Configures mergers to consume any data immediately.
  static void customizeInfrastructure(std::vector<framework::CompletionPolicy>&);

  static inline std::string mergerOutputBinding() { return "out"; };
  static inline std::string mergerIdString() { return "MERGER"; };
  static inline header::DataOrigin mergerDataOrigin() { return header::DataOrigin("MRGR"); };
  static inline header::DataDescription mergerDataDescription(std::string name)
  {
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  };
  static inline header::DataHeader::SubSpecificationType mergerSubSpec(size_t layer, size_t id)
  {
    return (layer << 16) + id;
  };

 private:
  std::string mName;
  size_t mId{0};
  size_t mLayer{1};
  framework::Inputs mInputSpecs;
  framework::OutputSpec mOutputSpec;
  MergerConfig mConfig;
};

} // namespace experimental::mergers
} // namespace o2

#endif //ALICEO2_MERGERBUILDER_H
