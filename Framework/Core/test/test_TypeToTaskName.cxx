// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <catch_amalgamated.hpp>
#include "Framework/AnalysisTask.h"
#include <string_view>

using namespace o2::framework;

TEST_CASE("TypeIdHelpers_BasicConversion")
{
  // Basic CamelCase to snake-case conversion
//  REQUIRE((type_to_task_name(std::string_view("SimpleTask")) == "simple-task"));
//  REQUIRE((type_to_task_name(std::string_view("MyTask")) == "my-task"));
//  REQUIRE((type_to_task_name(std::string_view("Task")) == "task"));
}

TEST_CASE("TypeIdHelpers_AbbreviationConsolidation")
{
  // Test ALICE detector abbreviations
//  REQUIRE(type_to_task_name(std::string_view("ITSQA")) == "its-qa");
//  REQUIRE(type_to_task_name(std::string_view("TPCQCTask")) == "tpc-qc-task");
  REQUIRE(type_to_task_name(std::string_view("EMCALQATask")) == "emcal-qa-task");
//  REQUIRE(type_to_task_name(std::string_view("HMPIDTask")) == "hmpid-task");
// REQUIRE(type_to_task_name(std::string_view("ITSTPCTask")) == "its-tpc-task");
// REQUIRE(type_to_task_name(std::string_view("QCFV0Task")) == "qc-fv0-task");
}

//TEST_CASE("TypeIdHelpers_QualityControlAbbreviations")
//{
//  // Test quality control abbreviations
//  REQUIRE(type_to_task_name(std::string_view("QATask")) == "qa-task");
//  REQUIRE(type_to_task_name(std::string_view("QCTask")) == "qc-task");
//  REQUIRE(type_to_task_name(std::string_view("QCDAnalysis")) == "qcd-analysis");
//}

TEST_CASE("TypeIdHelpers_ComplexNames")
{
  // Test complex combinations
//  REQUIRE(type_to_task_name(std::string_view("ITSQAAnalysisTask")) == "its-qa-analysis-task");
  REQUIRE(type_to_task_name(std::string_view("TPCEMCQCTask")) == "tpc-emc-qc-task");
//  REQUIRE(type_to_task_name(std::string_view("MyITSTask")) == "my-its-task");
}

//TEST_CASE("TypeIdHelpers_EdgeCases")
//{
//  // Single character
//  REQUIRE(type_to_task_name(std::string_view("A")) == "a");
//
//  // All uppercase. BC is Bunch Crossing!
//  //
//  REQUIRE(type_to_task_name(std::string_view("ABC")) == "a-bc");
//  REQUIRE(type_to_task_name(std::string_view("BC")) == "bc");
//
//  // Mixed with numbers (numbers are not uppercase, so no hyphens before them)
//  REQUIRE(type_to_task_name(std::string_view("Task123")) == "task123");
//}
