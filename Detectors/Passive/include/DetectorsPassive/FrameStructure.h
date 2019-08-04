// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// -------------------------------------------------------------------------
// -----             Implementation of the Frame structure             -----
// -----                Created 05/07/17  by S. Wenzel                 -----
// -----             Based on AliFRAMEv3 from AliRoot                  -----
// -------------------------------------------------------------------------

#ifndef O2_PASSIVE_FRAMESTRUCTURE_H
#define O2_PASSIVE_FRAMESTRUCTURE_H

#include <FairModule.h>

class TGeoCompositeShape;

namespace o2
{
namespace passive
{
// class supposed to provide the frame support structure common to TOF and TRD
class FrameStructure : public FairModule
{
 public:
  FrameStructure(const char* name, const char* title = "FrameStruct");

  /**  default constructor    */
  FrameStructure() = default;

  /**  destructor     */
  ~FrameStructure() override = default;

  /**  Clone this object (used in MT mode only)  */
  FairModule* CloneModule() const override;

  /**  Create the module geometry  */
  void ConstructGeometry() override;

  // query if constructed with holes
  bool hasHoles() const { return mWithHoles; }
  // set if to be constructed with/without holes
  void setHoles(bool flag) { mWithHoles = true; }

 private:
  /**  copy constructor (used in MT mode only)   */
  FrameStructure(const FrameStructure& rhs);

  void makeHeatScreen(const char* name, float dyP, int rot1, int rot2);
  void createWebFrame(const char* name, float dHz, float theta0, float phi0);
  void createMaterials();
  TGeoCompositeShape* createTOFRail(float y);

  bool mCaveIsAvailable = false; ///! if the mother volume is available (to hook the frame)

  //
  bool mWithHoles =
    true; //!< if holes are enabled (just a central place for other to query; no influence on frame structure)

  // medium IDs for the Frame
  int mAirMedID = -1;   //!
  int mSteelMedID = -1; //!
  int mAluMedID = -1;   //!
  int mG10MedID = -1;   //!

  ClassDefOverride(FrameStructure, 1);
};
} // namespace passive
} // namespace o2

#endif
