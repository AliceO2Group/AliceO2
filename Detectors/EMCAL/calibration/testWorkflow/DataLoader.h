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

#include <bitset>
#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gsl/span>

#include <DataFormatsEMCAL/Cell.h>
#include <DataFormatsEMCAL/CellCompressed.h>
#include <DataFormatsEMCAL/TriggerRecord.h>
#include <Framework/InputRecord.h>
#include <Framework/InputSpec.h>
#include <Framework/ProcessingContext.h>
#include <Headers/DataHeader.h>

namespace o2
{
namespace emcal
{

class DataLoader
{
 public:
  class ObjectTypeException : public std::exception
  {
   public:
    ObjectTypeException(const std::string_view request, const std::string_view expectType) : mRequest(request), mExpectType(expectType)
    {
      mMessage = "Invalid type for ";
      mMessage += mRequest.data();
      mMessage += ": expecting ";
      mMessage += expectType.data();
    }
    ~ObjectTypeException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    const char* getRequest() const noexcept
    {
      return mRequest.data();
    }

    const char* getExpectType() const noexcept
    {
      return mExpectType.data();
    }

   private:
    std::string mRequest;
    std::string mExpectType;
    std::string mMessage;
  };

  class ObjectRequestException : public std::exception
  {
   public:
    ObjectRequestException(const std::string_view request) : mRequest(request)
    {
      mMessage = "No load request for data type: ";
      mMessage += mRequest.data();
    }
    ~ObjectRequestException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    const char* getRequest() const noexcept
    {
      return mRequest.data();
    }

   private:
    std::string mRequest;
    std::string mMessage;
  };

  class UnsupportedRequestException : public std::exception
  {
    UnsupportedRequestException(const std::string_view request) : mRequest(request)
    {
      mMessage = "Request not supported by data loader: ";
      mMessage += mRequest.data();
    }
    ~UnsupportedRequestException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    const char* getRequest() const noexcept
    {
      return mRequest.data();
    }

   private:
    std::string mRequest;
    std::string mMessage;
  };

  enum Object_t {
    EMCCELL,
    EMCCELLCOMPRESSED,
    EMCCELLTRIGGERRECORD
  };

  enum class Request_t {
    CELL,
    CELLTRIGGERRECORD
  };

  static const char* getCellBinding() { return "emccells"; }
  static const char* getCellCompressedBinding() { return "emccellscompressed"; }
  static const char* getCellTriggerRecordBinding() { return "emccellstriggerrecords"; }

  DataLoader() = default;
  ~DataLoader() = default;

  void setLoadCells(bool doLoad = true) { mObjects.set(EMCCELL, doLoad); }
  void setLoadCompressedCells(bool doLoad = true) { mObjects.set(EMCCELLCOMPRESSED, doLoad); }
  void setLoadCellTriggerRecords(bool doLoad = true) { mObjects.set(EMCCELLTRIGGERRECORD, doLoad); }

  void defineInputs(std::vector<framework::InputSpec> inputs)
  {
    if (mObjects.test(EMCCELL)) {
      inputs.emplace_back(getCellBinding(), o2::header::gDataOriginEMC, "CELLS", 0, framework::Lifetime::Timeframe);
    }
    if (mObjects.test(EMCCELLCOMPRESSED)) {
      inputs.emplace_back(getCellCompressedBinding(), o2::header::gDataOriginEMC, "CELLS", 0, framework::Lifetime::Timeframe);
    }
    if (mObjects.test(EMCCELLTRIGGERRECORD)) {
      inputs.emplace_back(getCellTriggerRecordBinding(), o2::header::gDataOriginEMC, "CELLSTRGR", 0, framework::Lifetime::Timeframe);
    }
  }

  void updateObjects(framework::ProcessingContext& ctx)
  {
    if (mObjects.test(EMCCELL)) {
      auto cellinput = gsl::span<const o2::emcal::Cell>(*(ctx.inputs().get<gsl::span<const o2::emcal::Cell>>(getCellBinding())));
    }
    if (mObjects.test(EMCCELLCOMPRESSED)) {
      mCompressedCells = gsl::span<const o2::emcal::CellCompressed>(*(ctx.inputs().get<gsl::span<const o2::emcal::CellCompressed>>(getCellCompressedBinding())));
    }
    if (mObjects.test(EMCCELLTRIGGERRECORD)) {
      mCellTriggerRecords = gsl::span<const o2::emcal::TriggerRecord>(*(ctx.inputs().get<gsl::span<const o2::emcal::TriggerRecord>>(getCellTriggerRecordBinding())));
    }
  }

  template <typename objecttype>
  gsl::span<const objecttype> get(Request_t request)
  {
    switch (request) {
      case Request_t::CELL: {
        if (mObjects.test(EMCCELL)) {
          if constexpr (std::is_same<objecttype, o2::emcal::Cell>::value) {
            return mCells;
          } else {
            throw ObjectTypeException("Cell", "o2::emcal::Cell");
          }
        } else if (mObjects.test(EMCCELLCOMPRESSED)) {
          if constexpr (std::is_same<objecttype, o2::emcal::CellCompressed>::value) {
            return mCompressedCells;
          } else {
            throw ObjectTypeException("Cell", "o2::emcal::CellCompressed");
          }
        } else {
          throw ObjectRequestException("Cell");
        }
        break;
      }
      case Request_t::CELLTRIGGERRECORD: {
        if (mObjects.test(EMCCELLTRIGGERRECORD)) {
          if constexpr (std::is_same<objecttype, o2::emcal::TriggerRecord>::value) {
            return mCellTriggerRecords;
          } else {
            throw ObjectTypeException("CellTriggerRecord", "o2::emcal::TriggerRecord");
          }
        } else {
          throw ObjectRequestException("CellTriggerRecord");
        }
      }
    };
    throw ObjectRequestException("Unknown");
  }

 private:
  std::bitset<16> mObjects;
  gsl::span<const o2::emcal::Cell> mCells;
  gsl::span<const o2::emcal::CellCompressed> mCompressedCells;
  gsl::span<const o2::emcal::TriggerRecord> mCellTriggerRecords;

}; // namespace emcal

} // namespace emcal

} // namespace o2
