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

#ifndef EMCAL_PEDESTAL_PROCESSOR_DEVICE_H_
#define EMCAL_PEDESTAL_PROCESSOR_DEVICE_H_

#include <memory>
#include <vector>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "EMCALBase/Mapper.h"
#include "EMCALCalibration/PedestalProcessorData.h"

namespace o2::emcal
{

class Geometry;

/// \class PedestalProcessorDevice
/// \brief  Processor part of the EMCAL pedestal calibration workflow
/// \author Markus Fasel <markus.fasel@cern.ch.>, Oak Ridge National Laboratory
/// \ingroup EMCALCalib
/// \since March 21, 2024
class PedestalProcessorDevice : o2::framework::Task
{
 private:
  /// \class ModuleIndexException
  /// \brief Exception handling errors in calculation of the absolute module ID
  class ModuleIndexException : public std::exception
  {
   public:
    /// \enum ModuleType_t
    /// \brief Type of module raising the exception
    enum class ModuleType_t {
      CELL_MODULE,  ///< Cell module type
      LEDMON_MODULE ///< LEDMON module type
    };

    /// \brief Constructor for cell indices
    /// \param moduleIndex Index of the module raising the exception
    /// \param column Column of the cell
    /// \param row Row of the cell
    /// \param columnshifted Shifted column index
    /// \param rowshifted Shifted row index
    ModuleIndexException(int moduleIndex, int column, int row, int columnshifted, int rowshifted) : mModuleType(ModuleType_t::CELL_MODULE),
                                                                                                    mIndex(moduleIndex),
                                                                                                    mColumn(column),
                                                                                                    mRow(row),
                                                                                                    mColumnShifted(columnshifted),
                                                                                                    mRowShifted(rowshifted) {}

    /// \brief Constructor for LEDMON indices
    /// \param moduleIndex Index of the module raising the exception
    ModuleIndexException(int moduleIndex) : mModuleType(ModuleType_t::LEDMON_MODULE), mIndex(moduleIndex) {}

    /// \brief Destructor
    ~ModuleIndexException() noexcept final = default;

    /// \brief Access to error message
    /// \return Error message
    const char* what() const noexcept final { return "Invalid cell / LEDMON index"; }

    /// \brief Get type of module raising the exception
    /// \return Module type
    ModuleType_t getModuleType() const { return mModuleType; }

    /// \brief Get index of the module raising the exception
    /// \return Index of the module
    int getIndex() const { return mIndex; }

    /// \brief Get column raising the exception (cell-case)
    /// \return Column
    int getColumn() const { return mColumn; }

    /// \brief Get row raising the exception (cell-case)
    /// \return Row
    int getRow() const { return mRow; }

    /// \brief Get shifted column raising the exception (cell-case)
    /// \return Shifted column
    int getColumnShifted() const { return mColumnShifted; }

    /// \brief Get shifted row raising the exception (cell-case)
    /// \return Shifted row
    int getRowShifted() const { return mRowShifted; }

   private:
    ModuleType_t mModuleType; ///< Type of the module raising the exception
    int mIndex = -1;          ///< Index raising the exception
    int mColumn = -1;         ///< Column of the module (cell-case)
    int mRow = -1;            ///< Row of the module (cell-case)
    int mColumnShifted = -1;  ///< shifted column of the module (cell-case)
    int mRowShifted = -1;     /// << shifted row of the module (cell-case)
  };

  Geometry* mGeometry = nullptr;
  std::unique_ptr<MappingHandler> mMapper = nullptr;
  PedestalProcessorData mPedestalData;

 protected:
  int getCellAbsID(int supermoduleID, int column, int row) const;
  int geLEDMONAbsID(int supermoduleID, int moduleID) const;

 public:
  PedestalProcessorDevice() = default;
  ~PedestalProcessorDevice() final = default;

  void init(framework::InitContext& ctx) final;

  void run(framework::ProcessingContext& ctx) final;

  bool isLostTimeframe(framework::ProcessingContext& ctx) const;

  void sendData(framework::ProcessingContext& ctx, const PedestalProcessorData& data) const;
};

framework::DataProcessorSpec getPedestalProcessorDevice(bool askDistSTF);

} // namespace o2::emcal

#endif