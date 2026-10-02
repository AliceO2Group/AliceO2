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

#include "FITDCSMonitoring/FITDCSBaseConfigReader.h"

namespace o2::fit
{
rapidjson::Document FITDCSBaseConfigReader::parseJsonBuffer(gsl::span<const char> buffer, bool throwOnInvalidSchema)
{
  rapidjson::MemoryStream ms(buffer.data(), buffer.size());
  rapidjson::Document document;
  document.ParseStream(ms);

  std::string validationErrorMessage;
  if ((!throwOnInvalidSchema || validateSchema(document, validationErrorMessage)) == false) {
    std::string_view bufferView(buffer.data(), buffer.size());
    throw std::runtime_error("Received document does not match FEE configuration schema! Error message: " + validationErrorMessage);
  }

  return document;
}

void FITDCSBaseConfigReader::loadSchema(std::string_view schema)
{
  rapidjson::Document schemaDocument;
  schemaDocument.Parse(schema.data(), schema.size());
  if (schemaDocument.HasParseError()) {
    throw std::runtime_error("Cannot parse JSON schema");
  }
  mSchema = std::make_unique<rapidjson::SchemaDocument>(schemaDocument);
}

bool FITDCSBaseConfigReader::validateSchema(const rapidjson::Document& docs, std::string& errorMessage)
{
  rapidjson::SchemaValidator validator(*mSchema);
  if (docs.Accept(validator) == false) {
    rapidjson::StringBuffer buffer;
    std::ostringstream ss;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(buffer);
    ss << "Invalid schema: " << buffer.GetString() << '\n';
    ss << "Invalid keyword: " << validator.GetInvalidSchemaKeyword() << '\n';

    buffer.Clear();

    validator.GetInvalidDocumentPointer().StringifyUriFragment(buffer);
    ss << "Invalid document: " << buffer.GetString() << '\n';

    errorMessage = ss.str();
    return false;
  }
  return true;
}
} // namespace o2::fit