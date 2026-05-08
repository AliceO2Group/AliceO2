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

#include <catch_amalgamated.hpp>

#include "../src/DataInputDirector.h"

#include <TFile.h>
#include <TMap.h>
#include <TMemFile.h>
#include <TObjString.h>

using namespace o2::framework;

// Tests for DataInputDirectorContext::levelForOrigin

TEST_CASE("levelForOrigin empty mapping")
{
  DataInputDirectorContext ctx;
  CHECK(ctx.levelForOrigin("AOD") == -1);
  CHECK(ctx.levelForOrigin("DYN") == -1);
}

TEST_CASE("levelForOrigin single entry")
{
  DataInputDirectorContext ctx;
  ctx.parentLevelToOrigin = {{"DYN", 1}};
  CHECK(ctx.levelForOrigin("DYN") == 1);
  CHECK(ctx.levelForOrigin("AOD") == -1);
}

TEST_CASE("levelForOrigin multiple entries")
{
  DataInputDirectorContext ctx;
  ctx.parentLevelToOrigin = {{"DYN", 1}, {"EMB", 2}, {"EXT", 1}};
  CHECK(ctx.levelForOrigin("DYN") == 1);
  CHECK(ctx.levelForOrigin("EMB") == 2);
  CHECK(ctx.levelForOrigin("EXT") == 1);
  CHECK(ctx.levelForOrigin("AOD") == -1);
  CHECK(ctx.levelForOrigin("") == -1);
}

// Tests for DataInputDescriptor::navigateToLevel

TEST_CASE("navigateToLevel returns null with no input files")
{
  // With no input files setFile fails immediately → {nullptr, -1}
  DataInputDirectorContext ctx;
  ctx.allowedParentLevel = 2;
  DataInputDescriptor desc(false, 0, ctx);

  auto [parentFile, parentNumTF] = desc.navigateToLevel(0, 0, 1, "DYN");
  CHECK(parentFile == nullptr);
  CHECK(parentNumTF == -1);
}

// ---------------------------------------------------------------------------
// Helpers: build an AO2D-shaped TMemFile with one DF directory.
// The AO2D format uses top-level TDirectory entries named DF_<id>.
// An optional "parentFiles" TMap maps each DF name to its parent file path.
// ---------------------------------------------------------------------------

static TMemFile* makeAODFile(const char* name)
{
  auto* f = new TMemFile(name, "RECREATE");
  f->mkdir("DF_1");
  f->Write();
  return f;
}

static TMemFile* makeAODFileWithParent(const char* name, const char* parentName)
{
  auto* f = new TMemFile(name, "RECREATE");
  f->mkdir("DF_1");
  auto* parentMap = new TMap();
  parentMap->Add(new TObjString("DF_1"), new TObjString(parentName));
  parentMap->Write("parentFiles", TObject::kSingleKey);
  f->Write();
  return f;
}

TEST_CASE("navigateToLevel finds parent TMemFile")
{
  // child.root  DF_1  parentFiles: {DF_1 -> parent.root}
  // parent.root DF_1
  auto* parentMF = makeAODFile("parent.root");
  auto* childMF = makeAODFileWithParent("child.root", "parent.root");

  DataInputDirectorContext ctx;
  ctx.allowedParentLevel = 2;
  ctx.openFiles = {{"child.root", childMF}, {"parent.root", parentMF}};

  DataInputDescriptor desc(false, 0, ctx);
  desc.addFileNameHolder(makeFileNameHolder("child.root"));

  auto [parentDesc, parentNumTF] = desc.navigateToLevel(0, 0, 1, "AOD");

  REQUIRE(parentDesc != nullptr);
  // DF_1 is the only timeframe in the parent, so its index is 0
  CHECK(parentNumTF == 0);
}

TEST_CASE("navigateToLevel returns -1 for missing DF in parent")
{
  // child has DF_2 but parent only has DF_1 — findDFNumber returns -1
  auto* parentMF = makeAODFile("parent2.root");

  auto* childMF = new TMemFile("child2.root", "RECREATE");
  childMF->mkdir("DF_2");
  auto* parentMap = new TMap();
  parentMap->Add(new TObjString("DF_2"), new TObjString("parent2.root"));
  parentMap->Write("parentFiles", TObject::kSingleKey);
  childMF->Write();

  DataInputDirectorContext ctx;
  ctx.allowedParentLevel = 2;
  ctx.openFiles = {{"child2.root", childMF}, {"parent2.root", parentMF}};

  DataInputDescriptor desc(false, 0, ctx);
  desc.addFileNameHolder(makeFileNameHolder("child2.root"));

  auto [parentDesc, parentNumTF] = desc.navigateToLevel(0, 0, 1, "AOD");

  // Parent has DF_1 but child references DF_2 — not found in parent
  REQUIRE(parentDesc != nullptr);
  CHECK(parentNumTF == -1);
}
