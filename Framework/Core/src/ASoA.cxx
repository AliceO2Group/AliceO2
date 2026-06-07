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

#include "Framework/ASoA.h"
#include "ArrowDebugHelpers.h"
#include "Framework/RuntimeError.h"
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/config.h>
#include <TMemFile.h>
#include <TClass.h>
#include <TTree.h>
#include <TH1.h>
#include <TError.h>

namespace o2::soa
{
void accessingInvalidIndexFor(const char* getter)
{
  throw o2::framework::runtime_error_f("Accessing invalid index for %s", getter);
}
void dereferenceWithWrongType(const char* getter, const char* target)
{
  throw o2::framework::runtime_error_f("Trying to dereference index with a wrong type in %s_as<T> for base target \"%s\". Note that if you have several compatible index targets in your process() signature, the last one will be the one actually bound.", getter, target);
}
void missingFilterDeclaration(int hash, int ai)
{
  throw o2::framework::runtime_error_f("Null selection for %d (arg %d), missing Filter declaration?", hash, ai);
}

void getterNotFound(const char* targetColumnLabel)
{
  throw o2::framework::runtime_error_f("Getter for \"%s\" not found", targetColumnLabel);
}

void emptyColumnLabel()
{
  throw framework::runtime_error("columnLabel: must not be empty");
}

SelectionVector selectionToVector(gandiva::Selection const& sel)
{
  SelectionVector rows;
  rows.resize(sel->GetNumSlots());
  for (auto i = 0; i < sel->GetNumSlots(); ++i) {
    rows[i] = sel->GetIndex(i);
  }
  return rows;
}

SelectionVector sliceSelection(std::span<int64_t const> const& mSelectedRows, int64_t nrows, uint64_t offset)
{
  auto start = offset;
  auto end = start + nrows;
  auto start_iterator = std::lower_bound(mSelectedRows.begin(), mSelectedRows.end(), start);
  auto stop_iterator = std::lower_bound(start_iterator, mSelectedRows.end(), end);
  SelectionVector slicedSelection{start_iterator, stop_iterator};
  std::ranges::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                         [&start](int64_t idx) {
                           return idx - static_cast<int64_t>(start);
                         });
  return slicedSelection;
}

std::shared_ptr<arrow::Table> ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  bool notEmpty = (tables[0]->num_rows() != 0);
  std::ranges::for_each(tables, [&fields, &columns, notEmpty](auto const& t) {
    std::ranges::copy(t->fields(), std::back_inserter(fields));
    if (notEmpty) {
      std::ranges::copy(t->columns(), std::back_inserter(columns));
    }
  });
  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

namespace
{
template <typename T>
  requires(std::same_as<T, std::string>)
auto makeString(T const& str)
{
  return str.c_str();
}
template <typename T>
  requires(std::same_as<T, const char*>)
auto makeString(T const& str)
{
  return str;
}

template <typename T>
void canNotJoin(std::vector<std::shared_ptr<arrow::Table>> const& tables, std::span<T> labels)
{
  for (auto i = 0U; i < tables.size() - 1; ++i) {
    if (tables[i]->num_rows() != tables[i + 1]->num_rows()) {
      throw o2::framework::runtime_error_f("Tables %s and %s have different sizes (%d vs %d) and cannot be joined!",
                                           makeString(labels[i]), makeString(labels[i + 1]), tables[i]->num_rows(), tables[i + 1]->num_rows());
    }
  }
}
} // namespace

std::shared_ptr<arrow::Table> ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<const char* const> labels)
{
  if (tables.size() == 1) {
    return tables[0];
  }
  canNotJoin(tables, labels);
  return joinTables(std::forward<std::vector<std::shared_ptr<arrow::Table>>>(tables));
}

std::shared_ptr<arrow::Table> ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables, std::span<const std::string> labels)
{
  if (tables.size() == 1) {
    return tables[0];
  }
  canNotJoin(tables, labels);
  return joinTables(std::forward<std::vector<std::shared_ptr<arrow::Table>>>(tables));
}

std::shared_ptr<arrow::Table> ArrowHelpers::concatTables(std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  if (tables.size() == 1) {
    return tables[0];
  }
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  std::vector<std::shared_ptr<arrow::Field>> resultFields = tables[0]->schema()->fields();
  auto compareFields = [](std::shared_ptr<arrow::Field> const& f1, std::shared_ptr<arrow::Field> const& f2) {
    // Let's do this with stable sorting.
    return (!f1->Equals(f2)) && (f1->name() < f2->name());
  };
  for (size_t i = 1; i < tables.size(); ++i) {
    auto& fields = tables[i]->schema()->fields();
    std::vector<std::shared_ptr<arrow::Field>> intersection;

    std::set_intersection(resultFields.begin(), resultFields.end(),
                          fields.begin(), fields.end(),
                          std::back_inserter(intersection), compareFields);
    resultFields.swap(intersection);
  }

  for (auto& field : resultFields) {
    arrow::ArrayVector chunks;
    for (auto& table : tables) {
      auto ci = table->schema()->GetFieldIndex(field->name());
      if (ci == -1) {
        throw std::runtime_error("Unable to find field " + field->name());
      }
      auto column = table->column(ci);
      auto otherChunks = column->chunks();
      chunks.insert(chunks.end(), otherChunks.begin(), otherChunks.end());
    }
    columns.push_back(std::make_shared<arrow::ChunkedArray>(chunks));
  }

  return arrow::Table::Make(std::make_shared<arrow::Schema>(resultFields), columns);
}

// ASCII-only lowercase. Column labels are plain identifiers, so we deliberately
// avoid the locale-aware std::tolower: it goes through the C locale facet on
// every character and dominated getIndexFromLabel in profiles.
static constexpr char asciiToLower(char c)
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
}

arrow::ChunkedArray* getIndexFromLabel(arrow::Table* table, std::string_view label)
{
  // Take the exact-match common case first (string_view comparison checks length
  // then memcmp), and fall back to a case-insensitive scan only when the labels
  // differ in case.
  auto field = std::ranges::find_if(table->schema()->fields(), [label](std::shared_ptr<arrow::Field> const& f) {
    std::string_view name = f->name();
    return label == name ||
           std::ranges::equal(label, name, [](char c1, char c2) {
             return asciiToLower(c1) == asciiToLower(c2);
           });
  });
  if (field == table->schema()->fields().end()) {
    o2::framework::throw_error(o2::framework::runtime_error_f("Unable to find column with label %s.", label));
  }
  auto index = std::distance(table->schema()->fields().begin(), field);
  return table->column(index).get();
}

void notBoundTable(const char* tableName)
{
  throw o2::framework::runtime_error_f("Index pointing to %s is not bound! Did you subscribe to the table?", tableName);
}

void notFoundColumn(const char* label, const char* key)
{
  throw o2::framework::runtime_error_f(R"(Preslice not valid: table "%s" (or join based on it) does not have column "%s")", label, key);
}

void missingOptionalPreslice(const char* label, const char* key)
{
  throw o2::framework::runtime_error_f(R"(Optional Preslice with missing binding used: table "%s" (or join based on it) does not have column "%s")", label, key);
}

void* extractCCDBPayload(char* payload, size_t size, TClass const* cl, const char* what)
{
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  // does it have a flattened headers map attached in the end?
  TMemFile file("name", (char*)payload, size, "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (file.IsZombie()) {
    return nullptr;
  }

  if (!cl) {
    return nullptr;
  }
  auto object = file.GetObjectChecked(what, cl);
  if (!object) {
    // it could be that object was stored with previous convention
    // where the classname was taken as key
    std::string objectName(cl->GetName());
    objectName.erase(std::find_if(objectName.rbegin(), objectName.rend(), [](unsigned char ch) {
                       return !std::isspace(ch);
                     }).base(),
                     objectName.end());
    objectName.erase(objectName.begin(), std::find_if(objectName.begin(), objectName.end(), [](unsigned char ch) {
                       return !std::isspace(ch);
                     }));

    object = file.GetObjectChecked(objectName.c_str(), cl);
    LOG(warn) << "Did not find object under expected name " << what;
    if (!object) {
      return nullptr;
    }
    LOG(warn) << "Found object under deprecated name " << cl->GetName();
  }
  auto result = object;
  // We need to handle some specific cases as ROOT ties them deeply
  // to the file they are contained in
  if (cl->InheritsFrom("TObject")) {
    // make a clone
    // detach from the file
    auto tree = dynamic_cast<TTree*>((TObject*)object);
    if (tree) {
      tree->LoadBaskets(0x1L << 32); // make tree memory based
      tree->SetDirectory(nullptr);
      result = tree;
    } else {
      auto h = dynamic_cast<TH1*>((TObject*)object);
      if (h) {
        h->SetDirectory(nullptr);
        result = h;
      }
    }
  }
  return result;
}

std::function<framework::ConcreteDataMatcher(framework::ConcreteDataMatcher&&)> originReplacement(header::DataOrigin newOrigin)
{
  return [newOrigin](framework::ConcreteDataMatcher&& m) {
    if ((m.origin == header::DataOrigin{"AOD"}) && (newOrigin != header::DataOrigin{"AOD"})) {
      m.origin = newOrigin;
    }
    return m;
  };
}

} // namespace o2::soa

namespace o2::framework
{
std::string cutString(std::string&& str)
{
  auto pos = str.find('_');
  if (pos != std::string::npos) {
    str.erase(pos);
  }
  return str;
}

std::string strToUpper(std::string&& str)
{
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
  return str;
}

bool PreslicePolicyBase::isMissing() const
{
  return binding == "[MISSING]";
}

Entry const& PreslicePolicyBase::getBindingKey() const
{
  return bindingKey;
}

void PreslicePolicySorted::updateSliceInfo(SliceInfoPtr&& si)
{
  sliceInfo = si;
}

void PreslicePolicyGeneral::updateSliceInfo(SliceInfoUnsortedPtr&& si)
{
  sliceInfo = si;
}

std::shared_ptr<arrow::Table> PreslicePolicySorted::getSliceFor(int value, std::shared_ptr<arrow::Table> const& input, uint64_t& offset) const
{
  auto [offset_, count] = this->sliceInfo.getSliceFor(value);
  offset = static_cast<int64_t>(offset_);
  if (count == 0) {
    // Empty group: avoid slicing every column only to discard it. Cache one
    // empty (0-row) table per input table and reuse it (see GroupSlicer).
    if (emptySlice.first != input.get()) {
      emptySlice = {input.get(), input->Slice(0, 0)};
    }
    return emptySlice.second;
  }
  return input->Slice(offset_, count);
}

std::span<const int64_t> PreslicePolicyGeneral::getSliceFor(int value) const
{
  return this->sliceInfo.getSliceFor(value);
}
} // namespace o2::framework
