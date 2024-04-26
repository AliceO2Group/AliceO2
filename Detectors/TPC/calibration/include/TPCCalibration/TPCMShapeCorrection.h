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

/// \file TPCMShapeCorrection.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_TPCMSHAPECORRECTION
#define ALICEO2_TPC_TPCMSHAPECORRECTION

#include "DataFormatsTPC/Defs.h"
#include <vector>

class TTree;

namespace o2::tpc
{

/// arbitrary boundary potential at the IFC
struct BoundaryPotentialIFC {
  /// add boundary potential at given z coordinate
  void addValue(float potential, float z)
  {
    mPotential.emplace_back(potential);
    mZPosition.emplace_back(z);
  }
  std::vector<float> mPotential; ///< boundary potential
  std::vector<float> mZPosition; ///< z-position of the delta potential
  ClassDefNV(BoundaryPotentialIFC, 1);
};

/// struct containing all M-Shape boundary potentials
struct TPCMShape {
  /// get time of last stored boundary potential
  double getStartTimeMS() const { return mTimeMS.front(); }

  /// get time of first stored boundary potential
  double getEndTimeMS() const { return mTimeMS.back(); }

  /// adding boundary potential for given timestamp
  void addBoundaryPotentialIFC(const BoundaryPotentialIFC& pot, double time)
  {
    mTimeMS.emplace_back(time);
    mBoundaryPotentialIFC.emplace_back(pot);
  }

  std::vector<double> mTimeMS;                             ///< time of the boundary potential
  std::vector<BoundaryPotentialIFC> mBoundaryPotentialIFC; ///< definition of boundary potential
  int mZ{129};                                             ///< number of vertices for boundary potential in z direction
  int mR{129};                                             ///< number of vertices for boundary potential in radial direction
  int mPhi{5};                                             ///< number of vertices for boundary potential in phi direction
  int mBField{5};                                          ///< B-Field
  ClassDefNV(TPCMShape, 1);
};

/*
Class for storing the scalers (obtained from DCAs) which are used to correct for the M-shape distortions
*/

class TPCMShapeCorrection
{
 public:
  /// default constructor
  TPCMShapeCorrection() = default;

  /// \return returns run number for which this object is valid
  void setRun(int run) { mRun = run; }

  /// dump this object to a file
  /// \param file output file
  /// \param name name of the output object
  void dumpToFile(const char* file, const char* name);

  /// dump this object to several files each containing the correction for n minutes
  /// \param file output file
  /// \param nSlices number of slices
  void dumpToFileSlices(const char* file, const char* name, int nSlices);

  /// load from input file (which were written using the dumpToFile method)
  /// \param inpf input file
  /// \param name name of the object in the file
  void loadFromFile(const char* inpf, const char* name);

  /// set this object from input tree
  void setFromTree(TTree& tpcMShapeTree);

  /// set maximum delta time when searching for nearest boundary potential
  float setMaxDeltaTimeMS(float deltaTMS) { return mMaxDeltaTimeMS = deltaTMS; }

  /// adding M-shapes
  void setMShapes(const TPCMShape& mshapes) { mMShapes = mshapes; }

  /// \return returns run number for which this object is valid
  int getRun() const { return mRun; }

  /// \return returns M-shape boundary potential for given timestamp
  /// \param timestamp timestamp for which the boundary potential is queried
  BoundaryPotentialIFC getBoundaryPotential(const double timestamp) const;

  /// \return returns sampling time of the stored values
  float getMaxDeltaTimeMS() const { return mMaxDeltaTimeMS; }

  /// \return returns all stored M-shapes
  const auto& getMShapes() const { return mMShapes; }

 private:
  int mRun{};                 ///< run for which this object is valid
  float mMaxDeltaTimeMS = 50; ///< sampling time of the M-shape scalers
  TPCMShape mMShapes;         ///< vector containing the M-shapes

  ClassDefNV(TPCMShapeCorrection, 1);
};

} // namespace o2::tpc
#endif
