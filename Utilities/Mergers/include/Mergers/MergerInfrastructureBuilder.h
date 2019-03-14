// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_INFRASTRUCTUREBUILDER_H
#define ALICEO2_INFRASTRUCTUREBUILDER_H

/// \file MergerInfrastructureBuilder.h
/// \brief Definition of O2 MergerInfrastructureBuilder, v1.0
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerConfig.h"

#include <Framework/WorkflowSpec.h>

#include <string>

namespace o2
{
namespace experimental::mergers
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

  // todo: consider another interface - showing to the builder dataprocessorspecs which should be joined together
  void setInfrastructureName(std::string name);
  void setInputSpecs(const framework::Inputs& inputs);
  void setOutputSpec(const framework::OutputSpec& outputSpec);
  void setConfig(MergerConfig config);

  framework::WorkflowSpec generateInfrastructure();
  void generateInfrastructure(framework::WorkflowSpec&); // todo indicate that it throws

 private:
  std::string validateConfig();
  std::vector<size_t> computeNumberOfMergersPerLayer(const size_t inputs) const;

 private:
  std::string mInfrastructureName;
  framework::Inputs mInputs;
  framework::OutputSpec mOutputSpec;
  MergerConfig mConfig;
};

} // namespace experimental::mergers
} // namespace o2

#endif //ALICEO2_INFRASTRUCTUREBUILDER_H
