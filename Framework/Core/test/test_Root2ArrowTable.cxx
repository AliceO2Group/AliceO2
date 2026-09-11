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

#include "Framework/TableBuilder.h"
#include "Framework/ASoA.h"
#include "Framework/PluginManager.h"
#include "../src/ArrowDebugHelpers.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <TBufferFile.h>
#include <TClass.h>
#include <TDirectoryFile.h>
#include <TMemFile.h>
#include <TDirectory.h>
#include <TTree.h>
#include <TRandom.h>
#include <TFile.h>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <memory>

#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/buffer.h>
#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include "Framework/RootArrowFilesystem.h"

using namespace o2::framework;

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN_FULL(Px, px, float, "px");
DECLARE_SOA_COLUMN_FULL(Py, py, float, "py");
DECLARE_SOA_COLUMN_FULL(Pz, pz, float, "pz");
DECLARE_SOA_COLUMN_FULL(Xyz, xyz, float[3], "xyz");
DECLARE_SOA_COLUMN_FULL(Ij, ij, int[2], "ij");
DECLARE_SOA_COLUMN_FULL(Random, random, double, "random");
DECLARE_SOA_COLUMN_FULL(Ev, ev, int, "ev");
} // namespace test

DECLARE_SOA_TABLE(Test, "AOD", "ETAPHI",
                  test::Px, test::Py, test::Pz, test::Xyz, test::Ij,
                  test::Random, test::Ev);
} // namespace o2::aod

TEST_CASE("RootTree2Fragment")
{
  using namespace o2::framework;
  /// A directory holding a tree

  /// Create a simple TTree
  auto* file = new TBufferFile(TBuffer::kWrite);

  TTree t1("t1", "a simple Tree with simple variables");
  Float_t xyz[3];
  Int_t ij[2];
  Float_t px = 0, py = 1, pz = 2;
  Double_t random;
  Int_t ev;
  t1.Branch("px", &px, "px/F");
  t1.Branch("py", &py, "py/F");
  t1.Branch("pz", &pz, "pz/F");
  t1.Branch("random", &random, "random/D");
  t1.Branch("ev", &ev, "ev/I");
  t1.Branch("xyz", xyz, "xyz[3]/F");
  t1.Branch("ij", ij, "ij[2]/I");
  // fill the tree
  for (Int_t i = 0; i < 1000; i++) {
    xyz[0] = 1;
    xyz[1] = 2;
    xyz[2] = 3;
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    xyz[2] = i + 1;
    ij[0] = i;
    ij[1] = i + 1;
    random = gRandom->Rndm();
    ev = i + 1;
    t1.Fill();
  }
  file->WriteObjectAny(&t1, t1.Class());
  auto* fileRead = new TBufferFile(TBuffer::kRead, file->BufferSize(), file->Buffer(), false, nullptr);

  std::vector<char const*> capabilitiesSpecs = {
    "O2Framework:RNTupleObjectReadingCapability",
    "O2Framework:TTreeObjectReadingCapability",
  };

  std::vector<LoadablePlugin> plugins;
  for (auto spec : capabilitiesSpecs) {
    auto morePlugins = PluginManager::parsePluginSpecString(spec);
    for (auto& extra : morePlugins) {
      plugins.push_back(extra);
    }
  }
  REQUIRE(plugins.size() == 2);

  RootObjectReadingFactory factory;
  std::vector<char const*> configDiscoverySpec = {};
  PluginManager::loadFromPlugin<RootObjectReadingCapability, RootObjectReadingCapabilityPlugin>(plugins, factory.capabilities);
  REQUIRE(factory.capabilities.size() == 2);
  REQUIRE(factory.capabilities[0].name == "rntuple");
  REQUIRE(factory.capabilities[1].name == "ttree");

  // Plugins are hardcoded for now...
  auto format = factory.capabilities[1].factory().format();

  auto fs = std::make_shared<TBufferFileFS>(fileRead, factory);

  arrow::dataset::FileSource source("p", fs);
  REQUIRE(format->IsSupported(source) == true);
  auto schemaOpt = format->Inspect(source);
  REQUIRE(schemaOpt.ok());
  auto schema = *schemaOpt;
  REQUIRE(schema->num_fields() == 7);
  REQUIRE(schema->field(0)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(1)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(2)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(3)->type()->id() == arrow::float64()->id());
  REQUIRE(schema->field(4)->type()->id() == arrow::int32()->id());
  REQUIRE(schema->field(5)->type()->id() == arrow::fixed_size_list(arrow::float32(), 3)->id());
  REQUIRE(schema->field(6)->type()->id() == arrow::fixed_size_list(arrow::int32(), 2)->id());
  auto fragment = format->MakeFragment(source, {}, schema);
  REQUIRE(fragment.ok());
  auto options = std::make_shared<arrow::dataset::ScanOptions>();
  options->dataset_schema = schema;
  auto scanner = format->ScanBatchesAsync(options, *fragment);
  REQUIRE(scanner.ok());
  auto batches = (*scanner)();
  auto result = batches.result();
  REQUIRE(result.ok());
  REQUIRE((*result)->columns().size() == 7);
  REQUIRE((*result)->num_rows() == 1000);
}

bool validateContents(std::shared_ptr<arrow::RecordBatch> batch)
{
  {
    auto int_array = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("ev"));
    REQUIRE(int_array->length() == 100);
    for (int64_t j = 0; j < int_array->length(); j++) {
      REQUIRE(int_array->Value(j) == j + 1);
    }
  }

  {
    auto list_array = std::static_pointer_cast<arrow::FixedSizeListArray>(batch->GetColumnByName("xyz"));

    REQUIRE(list_array->length() == 100);
    // Iterate over the FixedSizeListArray
    for (int64_t i = 0; i < list_array->length(); i++) {
      auto value_slice = list_array->value_slice(i);
      auto float_array = std::static_pointer_cast<arrow::FloatArray>(value_slice);

      REQUIRE(float_array->Value(0) == 1);
      REQUIRE(float_array->Value(1) == 2);
      REQUIRE(float_array->Value(2) == i + 1);
    }
  }

  {
    auto list_array = std::static_pointer_cast<arrow::FixedSizeListArray>(batch->GetColumnByName("ij"));

    REQUIRE(list_array->length() == 100);
    // Iterate over the FixedSizeListArray
    for (int64_t i = 0; i < list_array->length(); i++) {
      auto value_slice = list_array->value_slice(i);
      auto int_array = std::static_pointer_cast<arrow::Int32Array>(value_slice);
      REQUIRE(int_array->Value(0) == i);
      REQUIRE(int_array->Value(1) == i + 1);
    }
  }

  {
    auto bool_array = std::static_pointer_cast<arrow::BooleanArray>(batch->GetColumnByName("bools"));

    REQUIRE(bool_array->length() == 100);
    for (int64_t j = 0; j < bool_array->length(); j++) {
      REQUIRE(bool_array->Value(j) == (j % 3 == 0));
    }
  }

  {
    auto list_array = std::static_pointer_cast<arrow::FixedSizeListArray>(batch->GetColumnByName("manyBools"));

    REQUIRE(list_array->length() == 100);
    for (int64_t i = 0; i < list_array->length(); i++) {
      auto value_slice = list_array->value_slice(i);
      auto bool_array = std::static_pointer_cast<arrow::BooleanArray>(value_slice);
      REQUIRE(bool_array->Value(0) == (i % 4 == 0));
      REQUIRE(bool_array->Value(1) == (i % 5 == 0));
    }
  }

  {
    auto list_array = std::static_pointer_cast<arrow::ListArray>(batch->GetColumnByName("vla"));

    REQUIRE(list_array->length() == 100);
    for (int64_t i = 0; i < list_array->length(); i++) {
      auto value_slice = list_array->value_slice(i);
      REQUIRE(value_slice->length() == (i % 10));
      auto int_array = std::static_pointer_cast<arrow::Int32Array>(value_slice);
      for (size_t j = 0; j < value_slice->length(); j++) {
        REQUIRE(int_array->Value(j) == j);
      }
    }
  }
  return true;
}

bool validateSchema(std::shared_ptr<arrow::Schema> schema)
{
  REQUIRE(schema->num_fields() == 11);
  REQUIRE(schema->field(0)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(1)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(2)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(3)->type()->id() == arrow::float64()->id());
  REQUIRE(schema->field(4)->type()->id() == arrow::int32()->id());
  REQUIRE(schema->field(5)->type()->id() == arrow::fixed_size_list(arrow::float32(), 3)->id());
  REQUIRE(schema->field(6)->type()->id() == arrow::fixed_size_list(arrow::int32(), 2)->id());
  REQUIRE(schema->field(7)->type()->id() == arrow::boolean()->id());
  REQUIRE(schema->field(8)->type()->id() == arrow::fixed_size_list(arrow::boolean(), 2)->id());
  REQUIRE(schema->field(9)->type()->id() == arrow::list(arrow::int32())->id());
  REQUIRE(schema->field(10)->type()->id() == arrow::int8()->id());
  return true;
}

bool validatePhysicalSchema(std::shared_ptr<arrow::Schema> schema)
{
  REQUIRE(schema->num_fields() == 12);
  REQUIRE(schema->field(0)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(0)->name() == "px");
  REQUIRE(schema->field(1)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(2)->type()->id() == arrow::float32()->id());
  REQUIRE(schema->field(3)->type()->id() == arrow::float64()->id());
  REQUIRE(schema->field(4)->type()->id() == arrow::int32()->id());
  REQUIRE(schema->field(5)->type()->id() == arrow::fixed_size_list(arrow::float32(), 3)->id());
  REQUIRE(schema->field(6)->type()->id() == arrow::fixed_size_list(arrow::int32(), 2)->id());
  REQUIRE(schema->field(7)->type()->id() == arrow::boolean()->id());
  REQUIRE(schema->field(8)->type()->id() == arrow::fixed_size_list(arrow::boolean(), 2)->id());
  REQUIRE(schema->field(9)->type()->id() == arrow::int32()->id());
  REQUIRE(schema->field(10)->type()->id() == arrow::list(arrow::int32())->id());
  REQUIRE(schema->field(11)->type()->id() == arrow::int8()->id());
  return true;
}

TEST_CASE("RootTree2Dataset")
{
  using namespace o2::framework;
  /// A directory holding a tree
  // auto *f = new TFile("Foo.root", "RECREATE");
  auto* f = new TMemFile("foo", "RECREATE");
  f->mkdir("DF_1");
  f->mkdir("DF_2");

  f->cd("DF_1");
  auto* t = new TTree("tracks", "a simple Tree with simple variables");
  {
    Float_t xyz[3];
    Int_t ij[2];
    Float_t px = 0, py = 1, pz = 2;
    Double_t random;
    Int_t ev;
    t->Branch("px", &px, "px/F");
    t->Branch("py", &py, "py/F");
    t->Branch("pz", &pz, "pz/F");
    t->Branch("random", &random, "random/D");
    t->Branch("ev", &ev, "ev/I");
    t->Branch("xyz", xyz, "xyz[3]/F");
    t->Branch("ij", ij, "ij[2]/I");
    // fill the tree
    for (Int_t i = 0; i < 1000; i++) {
      xyz[0] = 1;
      xyz[1] = 2;
      xyz[2] = 3;
      gRandom->Rannor(px, py);
      pz = px * px + py * py;
      xyz[2] = i + 1;
      ij[0] = i;
      ij[1] = i + 1;
      random = gRandom->Rndm();
      ev = i + 1;
      t->Fill();
    }
  }

  f->cd("DF_2");
  t = new TTree("tracks", "a simple Tree with simple variables");
  {
    Float_t xyz[3];
    Int_t ij[2];
    Float_t px = 0, py = 1, pz = 2;
    Double_t random;
    Int_t ev;
    bool oneBool;
    bool manyBool[2];
    int vla[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int vlaSize = 0;
    char byte;

    t->Branch("px", &px, "px/F");
    t->Branch("py", &py, "py/F");
    t->Branch("pz", &pz, "pz/F");
    t->Branch("random", &random, "random/D");
    t->Branch("ev", &ev, "ev/I");
    t->Branch("xyz", xyz, "xyz[3]/F");
    t->Branch("ij", ij, "ij[2]/I");
    t->Branch("bools", &oneBool, "bools/O");
    t->Branch("manyBools", &manyBool, "manyBools[2]/O");
    t->Branch("vla_size", &vlaSize, "vla_size/I");
    t->Branch("vla", vla, "vla[vla_size]/I");
    t->Branch("byte", &byte, "byte/B");
    // fill the tree
    for (Int_t i = 0; i < 100; i++) {
      xyz[0] = 1;
      xyz[1] = 2;
      xyz[2] = 3;
      gRandom->Rannor(px, py);
      pz = px * px + py * py;
      xyz[2] = i + 1;
      ij[0] = i;
      ij[1] = i + 1;
      random = gRandom->Rndm();
      ev = i + 1;
      oneBool = (i % 3 == 0);
      manyBool[0] = (i % 4 == 0);
      manyBool[1] = (i % 5 == 0);
      vlaSize = i % 10;
      byte = i;
      t->Fill();
    }
  }
  f->Write();

  std::vector<char const*> capabilitiesSpecs = {
    "O2Framework:RNTupleObjectReadingCapability",
    "O2Framework:TTreeObjectReadingCapability",
  };

  RootObjectReadingFactory factory;

  std::vector<LoadablePlugin> plugins;
  for (auto spec : capabilitiesSpecs) {
    auto morePlugins = PluginManager::parsePluginSpecString(spec);
    for (auto& extra : morePlugins) {
      plugins.push_back(extra);
    }
  }
  REQUIRE(plugins.size() == 2);

  PluginManager::loadFromPlugin<RootObjectReadingCapability, RootObjectReadingCapabilityPlugin>(plugins, factory.capabilities);

  REQUIRE(factory.capabilities.size() == 2);
  REQUIRE(factory.capabilities[0].name == "rntuple");
  REQUIRE(factory.capabilities[1].name == "ttree");

  // Plugins are hardcoded for now...
  auto rNtupleFormat = factory.capabilities[0].factory().format();
  auto format = factory.capabilities[1].factory().format();

  auto fs = std::make_shared<TFileFileSystem>(f, 50 * 1024 * 1024, factory);

  arrow::dataset::FileSource source("DF_2/tracks", fs);
  REQUIRE(format->IsSupported(source) == true);
  auto physicalSchema = format->Inspect(source);
  REQUIRE(physicalSchema.ok());
  REQUIRE(validatePhysicalSchema(*physicalSchema));
  // Create the dataset schema rather than using the physical one
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto& field : (*(physicalSchema))->fields()) {
    if (field->name().ends_with("_size")) {
      continue;
    }
    fields.push_back(field);
  }
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);

  validateSchema(schema);

  auto fragment = format->MakeFragment(source, {}, *physicalSchema);
  REQUIRE(fragment.ok());
  auto options = std::make_shared<arrow::dataset::ScanOptions>();
  options->dataset_schema = schema;
  auto scanner = format->ScanBatchesAsync(options, *fragment);
  REQUIRE(scanner.ok());

  // This is batch has deferred contents. Therefore we need to use a DeferredOutputStream to
  // write it to a real one and read it back with the BufferReader, which is hopefully zero copy
  std::shared_ptr<arrow::RecordBatch> batch;

  auto batches = (*scanner)();
  auto result = batches.result();
  REQUIRE(result.ok());
  REQUIRE((*result)->columns().size() == 11);
  REQUIRE((*result)->num_rows() == 100);
  std::shared_ptr<arrow::ResizableBuffer> buffer = *arrow::AllocateResizableBuffer(1000, 64);
  auto deferredWriterStream = factory.capabilities[1].factory().deferredOutputStreamer(*fragment, buffer);
  auto outBatch = arrow::ipc::MakeStreamWriter(deferredWriterStream.get(), schema);
  auto status = outBatch.ValueOrDie()->WriteRecordBatch(**result);
  std::shared_ptr<arrow::io::InputStream> bufferReader = std::make_shared<arrow::io::BufferReader>(buffer);
  auto readerResult = arrow::ipc::RecordBatchStreamReader::Open(bufferReader);
  auto batchReader = readerResult.ValueOrDie();

  auto next = batchReader->ReadNext(&batch);
  REQUIRE(batch != nullptr);

  validateContents(batch);

  auto* output = new TMemFile("foo", "RECREATE");
  auto outFs = std::make_shared<TFileFileSystem>(output, 0, factory);

  // Open a stream at toplevel
  auto destination = outFs->OpenOutputStream("/", {});
  REQUIRE(destination.ok());

  // Write to the /DF_3 tree at top level
  arrow::fs::FileLocator locator{outFs, "/DF_3"};
  auto writer = format->MakeWriter(*destination, schema, {}, locator);
  auto success = writer->get()->Write(batch);
  REQUIRE(batch->schema()->field(0)->name() == "px");
  auto rootDestination = std::dynamic_pointer_cast<TDirectoryFileOutputStream>(*destination);

  SECTION("Read tree")
  {
    REQUIRE(success.ok());
    // Let's read it back...
    auto tfileFs = std::dynamic_pointer_cast<TFileFileSystem>(outFs);
    REQUIRE(tfileFs.get());
    REQUIRE(tfileFs->GetFile());
    auto* tree = (TTree*)tfileFs->GetFile()->GetObjectChecked("/DF_3", TClass::GetClass("TTree"));
    REQUIRE(tree != nullptr);
    REQUIRE(((TBranch*)tree->GetListOfBranches()->At(0))->GetEntries() == 100);
    REQUIRE(((TBranch*)tree->GetListOfBranches()->At(0))->GetName() == std::string("px"));

    arrow::dataset::FileSource source2("/DF_3", outFs);

    REQUIRE(format->IsSupported(source2) == true);
    tfileFs = std::dynamic_pointer_cast<TFileFileSystem>(source2.filesystem());
    REQUIRE(tfileFs.get());
    REQUIRE(tfileFs->GetFile());
    REQUIRE(tfileFs->GetFile()->GetObjectChecked("/DF_3", TClass::GetClass("TTree")));

    tree = (TTree*)tfileFs->GetFile()->GetObjectChecked("/DF_3", TClass::GetClass("TTree"));
    REQUIRE(tree != nullptr);
    REQUIRE(((TBranch*)tree->GetListOfBranches()->At(0))->GetEntries() == 100);

    auto schemaOptWritten = format->Inspect(source2);
    tfileFs = std::dynamic_pointer_cast<TFileFileSystem>(source2.filesystem());
    REQUIRE(tfileFs.get());
    REQUIRE(tfileFs->GetFile());
    REQUIRE(tfileFs->GetFile()->GetObjectChecked("/DF_3", TClass::GetClass("TTree")));
    REQUIRE(schemaOptWritten.ok());
    auto schemaWritten = *schemaOptWritten;

    tree = (TTree*)tfileFs->GetFile()->GetObjectChecked("/DF_3", TClass::GetClass("TTree"));
    REQUIRE(tree != nullptr);
    REQUIRE(((TBranch*)tree->GetListOfBranches()->At(0))->GetEntries() == 100);

    REQUIRE(validatePhysicalSchema(schemaWritten));
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto& field : schemaWritten->fields()) {
      if (field->name().ends_with("_size")) {
        continue;
      }
      fields.push_back(field);
    }
    std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);
    REQUIRE(validateSchema(schema));

    auto fragmentWritten = format->MakeFragment(source2, {}, *physicalSchema);
    REQUIRE(fragmentWritten.ok());
    auto optionsWritten = std::make_shared<arrow::dataset::ScanOptions>();
    optionsWritten->dataset_schema = schema;
    auto scannerWritten = format->ScanBatchesAsync(optionsWritten, *fragmentWritten);
    REQUIRE(scannerWritten.ok());
    tree = (TTree*)tfileFs->GetFile()->GetObjectChecked("/DF_3", TClass::GetClass("TTree"));
    REQUIRE(tree != nullptr);
    REQUIRE(((TBranch*)tree->GetListOfBranches()->At(0))->GetEntries() == 100);
    auto batchesWritten = (*scannerWritten)();
    auto resultWritten = batchesWritten.result();
    REQUIRE(resultWritten.ok());
    REQUIRE((*resultWritten)->columns().size() == 11);
    REQUIRE((*resultWritten)->num_rows() == 100);

    std::shared_ptr<arrow::ResizableBuffer> buffer = *arrow::AllocateResizableBuffer(1000, 64);
    auto deferredWriterStream2 = factory.capabilities[1].factory().deferredOutputStreamer(*fragmentWritten, buffer);
    auto outBatch = arrow::ipc::MakeStreamWriter(deferredWriterStream2.get(), schema);
    auto status = outBatch.ValueOrDie()->WriteRecordBatch(**resultWritten);
    std::shared_ptr<arrow::io::InputStream> bufferReader = std::make_shared<arrow::io::BufferReader>(buffer);
    auto readerResult = arrow::ipc::RecordBatchStreamReader::Open(bufferReader);
    auto batchReader = readerResult.ValueOrDie();

    auto next = batchReader->ReadNext(&batch);
    REQUIRE(batch != nullptr);
    validateContents(batch);
  }

#if __has_include(<ROOT/RFieldBase.hxx>)
  arrow::fs::FileLocator rnTupleLocator{outFs, "rntuple"};
#else
  arrow::fs::FileLocator rnTupleLocator{outFs, "/rntuple"};
#endif
  // We write an RNTuple in the same TMemFile, using /rntuple as a location
  auto rntupleDestination = std::dynamic_pointer_cast<TDirectoryFileOutputStream>(*destination);

  {
    auto rNtupleWriter = rNtupleFormat->MakeWriter(*destination, schema, {}, rnTupleLocator);
    auto rNtupleSuccess = rNtupleWriter->get()->Write(batch);
    REQUIRE(rNtupleSuccess.ok());
  }

  // And now we can read back the RNTuple into a RecordBatch
#if __has_include(<ROOT/RFieldBase.hxx>)
  arrow::dataset::FileSource writtenRntupleSource("rntuple", outFs);
#else
  arrow::dataset::FileSource writtenRntupleSource("/rntuple", outFs);
#endif

  REQUIRE(rNtupleFormat->IsSupported(writtenRntupleSource) == true);

  auto rntupleSchemaOpt = rNtupleFormat->Inspect(writtenRntupleSource);
  REQUIRE(rntupleSchemaOpt.ok());
  auto rntupleSchemaWritten = *rntupleSchemaOpt;
  REQUIRE(validateSchema(rntupleSchemaWritten));

  auto rntupleFragmentWritten = rNtupleFormat->MakeFragment(writtenRntupleSource, {}, rntupleSchemaWritten);
  REQUIRE(rntupleFragmentWritten.ok());
  auto rntupleOptionsWritten = std::make_shared<arrow::dataset::ScanOptions>();
  rntupleOptionsWritten->dataset_schema = rntupleSchemaWritten;
  auto rntupleScannerWritten = rNtupleFormat->ScanBatchesAsync(rntupleOptionsWritten, *rntupleFragmentWritten);
  REQUIRE(rntupleScannerWritten.ok());
  auto rntupleBatchesWritten = (*rntupleScannerWritten)();
  auto rntupleResultWritten = rntupleBatchesWritten.result();
  REQUIRE(rntupleResultWritten.ok());
  REQUIRE((*rntupleResultWritten)->columns().size() == 11);
  REQUIRE(validateSchema((*rntupleResultWritten)->schema()));
  REQUIRE((*rntupleResultWritten)->num_rows() == 100);
  REQUIRE(validateContents(*rntupleResultWritten));
}
