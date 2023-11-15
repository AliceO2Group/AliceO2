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

#ifndef ALICEO2_INFRASTRUCTUREBUILDER_H
#define ALICEO2_INFRASTRUCTUREBUILDER_H

/// \file MergerInfrastructureBuilder.h
/// \brief Definition of O2 MergerInfrastructureBuilder, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerConfig.h"

#include <Framework/WorkflowSpec.h>

#include <string>

namespace o2::mergers
{

/// \brief Builder class for Merger topologies
///
/// This class is supposed to be the default merger topology generation interface.
/// Use it by creating an MergerInfrastructureBuilder object, setting the infrastructure name,
/// InputSpecs, OutputSpec and configuration. Generate the DataProcessorSpecs by using
/// generateInfrastructure(), which will throw if the configuration is invalid. See
/// MergerConfig.h for configuration options.
/// When creating a topology, do not forget to configure the completion policies by placing following customization
/// function before the inclusion of `Framework/runDataProcessing.h`
/// \code{.cxx}
/// void customize(std::vector<CompletionPolicy>& policies)
/// {
///   MergerBuilder::customizeInfrastructure(policies);
/// }
/// \endcode
class MergerInfrastructureBuilder
{

 public:
  /// \brief Default constructor.
  MergerInfrastructureBuilder();
  /// \brief Default destructor.
  ~MergerInfrastructureBuilder() = default;

  void setInfrastructureName(std::string name);
  void setInputSpecs(const framework::Inputs& inputs);
  void setOutputSpec(const framework::OutputSpec& outputSpec);
  void setOutputSpecMovingWindow(const framework::OutputSpec& outputSpec);
  void setConfig(MergerConfig config);

  framework::WorkflowSpec generateInfrastructure();
  void generateInfrastructure(framework::WorkflowSpec&); // todo indicate that it throws

 private:
  std::string validateConfig();
  std::vector<size_t> computeNumberOfMergersPerLayer(const size_t inputs) const;

 private:
  std::string mInfrastructureName;
  framework::Inputs mInputs;
  framework::OutputSpec mOutputSpecIntegral;
  framework::OutputSpec mOutputSpecMovingWindow;
  MergerConfig mConfig;
};

} // namespace o2::mergers

#endif //ALICEO2_INFRASTRUCTUREBUILDER_H
