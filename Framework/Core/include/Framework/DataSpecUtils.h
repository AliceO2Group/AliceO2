// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATASPECUTILS_H
#define FRAMEWORK_DATASPECUTILS_H

#include "Framework/OutputSpec.h"
#include "Framework/InputSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/Output.h"

#include <optional>

namespace o2
{
namespace framework
{

struct DataSpecUtils {
  /// @return true if a given InputSpec @a spec matches with a @a target ConcreteDataMatcher
  static bool match(InputSpec const& spec, ConcreteDataMatcher const& target);

  /// @return true if a given InputSpec @a input  matches the @a output outputspec
  static bool match(InputSpec const& input, OutputSpec const& output);

  static bool match(const InputSpec& spec,
                    const o2::header::DataOrigin& origin,
                    const o2::header::DataDescription& description,
                    const o2::header::DataHeader::SubSpecificationType& subSpec);

  static bool match(const OutputSpec &spec,
                    const o2::header::DataOrigin &origin,
                    const o2::header::DataDescription &description,
                    const o2::header::DataHeader::SubSpecificationType &subSpec) {
    return spec.origin == origin &&
           spec.description == description &&
           spec.subSpec == subSpec;
  }

  template <typename T>
  static bool match(const T&spec, const o2::header::DataHeader &header) {
    return DataSpecUtils::match(spec,
                                header.dataOrigin,
                                header.dataDescription,
                                header.subSpecification);
  }

  /// Describes an InputSpec. Use this to get some human readable
  /// version of the contents of the InputSpec. Notice this is not part
  /// of the InputSpec API, because there is no unique way a description should
  /// be done, so we keep this outside.
  static std::string describe(InputSpec const& spec);

  /// Provide a unique label for the input spec. Again this is outside because there
  /// is no standard way of doing it, so better not to pollute the API.
  static std::string label(InputSpec const& spec);

  /// Provides the to be used as suffix for any REST endpoint related
  /// to the @a spec.
  static std::string restEndpoint(InputSpec const& spec);

  /// Given an InputSpec, manipulate it so that it will match only the given
  /// subSpec.
  static void updateMatchingSubspec(InputSpec& in, header::DataHeader::SubSpecificationType subSpec);
  /// Given an OutputSpec, manipulate it so that it will match only the given
  /// subSpec.
  static void updateMatchingSubspec(OutputSpec& in, header::DataHeader::SubSpecificationType subSpec);

  /// Validates the given InputSpec @a in
  static bool validate(InputSpec const& in);

  /// Same as the other describe, but uses a buffer to reduce memory churn.
  static void describe(char* buffer, size_t size, InputSpec const& spec);

  /// If possible extract the ConcreteDataMatcher from an InputSpec. This
  /// can be done either if the InputSpec is defined in terms for a ConcreteDataMatcher
  /// or if the query can be uniquely assigned to a ConcreteDataMatcher.
  static ConcreteDataMatcher asConcreteDataMatcher(InputSpec const& input);

  /// If possible extract the ConcreteDataMatcher from an OutputSpec. 
  /// For the moment this is trivial as the OutputSpec does not allow
  /// for wildcards.
  static ConcreteDataMatcher asConcreteDataMatcher(OutputSpec const& spec);

  /// Get the subspec, if available.
  static std::optional<header::DataHeader::SubSpecificationType> getOptionalSubSpec(OutputSpec const& spec);
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_DATASPECUTILS_H
