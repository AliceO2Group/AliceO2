// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_CLUSTER_H_
#define ALICEO2_EMCAL_CLUSTER_H_

#include <array>
#include <exception>
#include <iosfwd>
#include <string>
#include <vector>
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "MathUtils/Cartesian3D.h"

namespace o2
{

namespace emcal
{

/// \class Cluster
/// \brief EMCAL Cluster
///
class Cluster : public o2::dataformats::TimeStamp<Float16_t>, public Point3D<Float16_t>
{
  using CellIndexRange = o2::dataformats::RangeRefComp<8>;

 public:
  class InvalidPositionException : public std::exception
  {
   public:
    InvalidPositionException() = default;
    InvalidPositionException(const std::array<Float_t, 3>& pos) : std::exception(), mPosition(pos) {}
    ~InvalidPositionException() noexcept override = default;

    const char* what() const noexcept override
    {
      return Form("Invalid position: %f, %f, %f", mPosition[0], mPosition[1], mPosition[2]);
    }

   private:
    std::array<Float_t, 3> mPosition;
    std::string mMessage;
  };

  Cluster() = default;
  Cluster(Point3D<Float_t> pos, Float_t time, Float_t energy);
  Cluster(Float_t x, Float_t y, Float_t z, Float_t time, Float_t energy);
  ~Cluster() noexcept = default;

  Float_t getM02() const { return mM02; }
  Float_t getM20() const { return mM20; }
  Float_t getDispersion() const { return mDispersion; }
  Float_t getDistanceToBadCell() const { return mDistanceToBadCell; }
  Bool_t isExotic() const { return mIsExotic; }
  Int_t getNCells() const { return mCellIndices.getEntries(); }
  Int_t getCellIndexFirst() const { return mCellIndices.getFirstEntry(); }
  CellIndexRange getCellIndexRange() const { return mCellIndices; }

  void setIsExotic(Bool_t exotic) { mIsExotic = exotic; }
  void setM02(Float_t val) { mM02 = val; }
  void setM20(Float_t val) { mM20 = val; }
  void setDispersion(Float_t val) { mDispersion = val; }
  void setDistanceToBadCell(Float_t distance) { mDistanceToBadCell = distance; }
  void setCellIndices(int firstcell, int ncells)
  {
    mCellIndices.setFirstEntry(firstcell);
    mCellIndices.setEntries(ncells);
  }
  void setFirstCell(int firstcell) { mCellIndices.setFirstEntry(firstcell); }
  void setNCells(int ncells) { mCellIndices.setEntries(ncells); }

  Vector3D<Float_t> getMomentum(const Point3D<Float_t>* vertex = nullptr) const;

  void PrintStream(std::ostream& stream) const;

 private:
  Float16_t mEnergy;            ///< Cluster energy
  Float16_t mDispersion;        ///< Cluster dispersion
  Float16_t mDistanceToBadCell; ///< Distance to nearest bad cell
  Float16_t mM02;               ///< M02 parameter (2-nd moment along the main eigen axis.)
  Float16_t mM20;               ///< M20 parameter (2-nd moment along the second eigen axis.)
  Bool_t mIsExotic;             ///< Mark cluster as exotic cluster
  CellIndexRange mCellIndices;  ///< Cells contributing to a cluser

  ClassDefNV(Cluster, 1);
};

std::ostream& operator<<(std::ostream& stream, const o2::emcal::Cluster& cluster);

} // namespace emcal

} // namespace o2

#endif
