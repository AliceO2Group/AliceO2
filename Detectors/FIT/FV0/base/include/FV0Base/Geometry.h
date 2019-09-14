// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Geometry.h
/// \brief  Base definition of FV0 geometry.
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland
/// \author Andreas Molander, University of Helsinki, Finland

#ifndef ALICEO2_FV0_GEOMETRY_H_
#define ALICEO2_FV0_GEOMETRY_H_

#include <vector>

#include <TGeoMatrix.h>
#include <TGeoVolume.h>
#include <TVirtualMC.h>

namespace o2
{
namespace fv0
{
/// FV0 Geometry
class Geometry
{
 public:
  /// Geometry type options possible to be initialized. The type of the geometry will specify which components are
  /// created.
  enum EGeoType {
    eUninitilized,
    eOnlySensitive,
    eRough,
    eFull
  };

  /// Geometry components possible to be enabled/disabled. Only enabled components will be created.
  enum EGeoComponent {
    eScintillator,
    ePlastics,
    eFibers,
    eScrews,
    eRods,
    eContainer
  };

  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  Geometry() : mGeometryType(eUninitilized), mLeftTransformation(nullptr), mRightTransformation(nullptr){};

  /// Standard constructor
  /// \param initType[in]  The type of geometry, that will be initialized
  ///                       -> initType == eUnitialized   => no parts
  ///                       -> initType == eOnlySensitive => only sensitive detector parts
  ///                       -> initType == eRough         => sensitive parts and rough structural elements
  ///                       -> initType == eFull          => complete, detailed geometry (including screws, etc.)
  /// \return  -
  explicit Geometry(EGeoType initType);

  /// Copy constructor.
  Geometry(const Geometry& geometry);

  /// Destructor
  ~Geometry();

  /// Get the unique ID of the current scintillator cell during simulation.
  /// The ID is a number starting from 0 at the first cell right of the y-axis
  /// and continues clockwise one ring at a time.
  /// \param  fMC The virtual Monte Carlo interface.
  /// \return The ID of the current scintillator cell during simulation.
  const int getCurrentCellId(const TVirtualMC* fMC) const;

  /// Get the names of all the sensitive volumes of the geometry.
  /// \return The names of all the sensitive volumes of the geometry.
  const std::vector<std::string>& getSensitiveVolumeNames() const { return mSensitiveVolumeNames; };

  /// Enable or disable a geometry component. To be called before the geometry is built. A disabled component will not
  /// be added to the geometry. The enabled components are by default specified by the geometry type.
  /// \param  component   The geometry component to be enabled/disabled.
  /// \param  enable      Setting the enabled state. Default is true.
  /// \return The enabled state of the geometry component.
  const bool enableComponent(EGeoComponent component, bool enable = true);

  /// Build the geometry.
  void buildGeometry() const;

 private:
  inline static const std::string sDetectorName = "FV0";

  // General geometry constants
  static constexpr float sEpsilon = 0.01;     ///< Used to make one spatial dimension infinitesimally larger than other
  static constexpr float sDzScintillator = 4; ///< Thickness of the scintillator
  static constexpr float sDzPlastic = 1;      ///< Thickness of the fiber plastics

  static constexpr float sXGlobal = 0; ///< Global x-position of the geometrical center of scintillators
  static constexpr float sYGlobal = 0; ///< Global y-position of the geometrical center of scintillators
  // FT0 starts at z=328
  static constexpr float sZGlobal = 320 - sDzScintillator / 2; ///< Global z-pos of geometrical center of scintillators
  static constexpr float sDxHalvesSeparation = 0;              ///< Separation between the left and right side of the detector
  static constexpr float sDyHalvesSeparation = 0;              ///< y-position of the right detector part relative to the left part
  static constexpr float sDzHalvesSeparation = 0;              ///< z-position of the right detector part relative to the left part

  /// Cell and scintillator constants
  static constexpr int sNumberOfCellSectors = 4; ///< Number of cell sectors for one half of the detector
  static constexpr int sNumberOfCellRings = 5;   ///< Number of cell rings
  /// Average cell ring radii.
  static constexpr float sCellRingRadii[sNumberOfCellRings + 1]{4.01, 7.3, 12.9, 21.25, 38.7, 72.115};
  static constexpr char sCellTypes[sNumberOfCellSectors]{'a', 'b', 'b', 'a'}; ///< Ordered cell types per half a ring
  /// Separation between the scintillator cells; paint thickness + half of separation gap.
  static constexpr float sDrSeparationScint = 0.03 + 0.04;

  /// Shift of the inner radius origin of the scintillators.
  static constexpr float sXShiftInnerRadiusScintillator = -0.15;
  /// Extension of the scintillator holes for the metal rods
  static constexpr float sDxHoleExtensionScintillator = 0.2;
  static constexpr float sDrHoleSmallScintillator = 0.265; ///< Radius of the small scintillator screw hole
  static constexpr float sDrHoleLargeScintillator = 0.415; ///< Radius of the large scintillator screw hole

  // Container constants
  static constexpr float sDzContainer = 30;                  ///< Depth of the metal container
  static constexpr float sDrContainerHole = 4.05;            ///< Radius of the beam hole in the metal container
  static constexpr float sXShiftContainerHole = -0.15;       ///< x-shift of the beam hole in the metal container
  static constexpr float sDrMaxContainerBack = 83.1;         ///< Outer radius of the container backplate
  static constexpr float sDzContainerBack = 1;               ///< Thickness of the container backplate
  static constexpr float sDrMinContainerFront = 45.7;        ///< Inner radius of the container frontplate
  static constexpr float sDrMaxContainerFront = 83.1;        ///< Outer radius of the container frontplate
  static constexpr float sDzContainerFront = 1;              ///< Thickness of the container frontplate
  static constexpr float sDxContainerStand = 40;             ///< Width of the container stand
  static constexpr float sDyContainerStand = 3;              ///< Height of the container stand at its center in x
  static constexpr float sDrMinContainerCone = 24.3;         ///< Inner radius at bottom of container frontplate cone
  static constexpr float sDzContainerCone = 16.2;            ///< Depth of the container frontplate cone
  static constexpr float sThicknessContainerCone = 0.6;      ///< Thickness of the container frontplate cone
  static constexpr float sXYThicknessContainerCone = 0.975;  ///< Radial thickness in the xy-plane of container cone
  static constexpr float sDrMinContainerOuterShield = 82.5;  ///< Inner radius of outer container shield
  static constexpr float sDrMaxContainerOuterShield = 82.65; ///< Outer radius of outer container shield
  static constexpr float sDrMinContainerInnerShield = 4;     ///< Inner radius of the inner container shield
  static constexpr float sDrMaxContainerInnerShield = 4.05;  ///< Outer radius of inner container shield
  static constexpr float sDxContainerCover = 0.15;           ///< Thickness of the container cover
  static constexpr float sDxContainerStandBottom = 38.5;     ///< Width of the bottom of the container stand
  static constexpr float sDyContainerStandBottom = 2;        ///< Thickness of the bottom of the container stand

  // Local position constants

  /// x-position of the right half of the scintillator.
  static constexpr float sXScintillator = sDxContainerCover;
  /// z-position of the scintillator cells.
  static constexpr float sZScintillator = 0;
  /// z-position of the plastic cells.
  static constexpr float sZPlastic = sZScintillator + sDzScintillator / 2 + sDzPlastic / 2;
  /// z-position of the container backplate.
  static constexpr float sZContainerBack = sZScintillator - sDzScintillator / 2 - sDzContainerBack / 2;
  /// z-position of the container frontplate.
  static constexpr float sZContainerFront = sZContainerBack - sDzContainerBack / 2 + sDzContainer - sDzContainerFront / 2;
  /// z-position of the center of the container.
  static constexpr float sZContainerMid = (sZContainerBack + sZContainerFront) / 2;
  /// z-position of the fiber volumes.
  static constexpr float sZFiber = (sZPlastic + sZContainerFront) / 2;
  /// z-position of the container frontplate cone.
  static constexpr float sZCone = sZContainerFront + sDzContainerFront / 2 - sDzContainerCone / 2;
  /// x shift of all screw holes.
  static constexpr float sXShiftScrews = sXScintillator;

  // Screw and rod dimensions

  /// Number of the different screw types.
  static constexpr int sNumberOfScrewTypes = 6;
  /// Radii of the thinner part of the screw types.
  static constexpr float sDrMinScrewTypes[sNumberOfScrewTypes]{0.25, 0.25, 0.4, 0.4, 0.4, 0.4};
  /// Radii of the thicker part of the screw types.
  static constexpr float sDrMaxScrewTypes[sNumberOfScrewTypes]{0, 0.5, 0.6, 0.6, 0.6, 0};
  /// Length of the thinner part of the screw types.
  static constexpr float sDzMaxScrewTypes[sNumberOfScrewTypes]{6.02, 13.09, 13.1, 23.1, 28.3, 5};
  /// Length of the thicker part of the screw types.
  static constexpr float sDzMinScrewTypes[sNumberOfScrewTypes]{0, 6.78, 6.58, 15.98, 21.48, 0};
  /// z shift of the screws. 0 means they are aligned with the scintillator.
  static constexpr float sZShiftScrew = 0;

  /// Number of the different rod types.
  static constexpr int sNumberOfRodTypes = 4;
  /// Width of the thinner part of the rod types.
  static constexpr float sDxMinRodTypes[sNumberOfRodTypes]{0.366, 0.344, 0.344, 0.344};
  /// Width of the thicker part of the rod types.
  static constexpr float sDxMaxRodTypes[sNumberOfRodTypes]{0.536, 0.566, 0.566, 0.716};
  /// Height of the thinner part of the rod types.
  static constexpr float sDyMinRodTypes[sNumberOfRodTypes]{0.5, 0.8, 0.8, 0.8};
  /// Height of the thicker part of the rod types.
  static constexpr float sDyMaxRodTypes[sNumberOfRodTypes]{0.9, 1.2, 1.2, 1.2};
  /// Length of the thinner part of the rod types.
  static constexpr float sDzMaxRodTypes[sNumberOfRodTypes]{12.5, 12.5, 22.5, 27.7};
  /// Length of the thicker part of the rod types.
  static constexpr float sDzMinRodTypes[sNumberOfRodTypes]{7.45, 7.45, 17.45, 22.65};
  /// z shift of the rods. 0 means they are aligned with tht scintillators.
  static constexpr float sZShiftRod = -0.05;

  // Strings for volume names, etc.
  inline static const std::string sScintillatorName = "SCINT";
  inline static const std::string sPlasticName = "PLAST";
  inline static const std::string sSectorName = "SECTOR";
  inline static const std::string sCellName = "CELL";
  inline static const std::string sScintillatorSectorName = sScintillatorName + sSectorName;
  inline static const std::string sScintillatorCellName = sScintillatorName + sCellName;
  inline static const std::string sPlasticSectorName = sPlasticName + sSectorName;
  inline static const std::string sPlasticCellName = sPlasticName + sCellName;
  inline static const std::string sFiberName = "FIBER";
  inline static const std::string sScrewName = "SCREW";
  inline static const std::string sScrewHolesCSName = "FV0SCREWHOLES";
  inline static const std::string sRodName = "ROD";
  inline static const std::string sRodHolesCSName = "FV0RODHOLES";
  inline static const std::string sContainerName = "CONTAINER";

  /// Initialize the geometry.
  void initializeGeometry();

  /// Initialize maps with geometry information.
  void initializeMaps();

  /// Initialize vectors with geometry information.
  void initializeVectors();

  /// Initialize common transformations.
  void initializeTransformations();

  /// Initialize the cell ring radii.
  void initializeCellRingRadii();

  /// Initialize sector transformations.
  void initializeSectorTransformations();

  /// Initialize fiber volume radii.
  void initializeFiberVolumeRadii();

  /// Initialize fiber mediums.
  void initializeFiberMedium();

  /// Initialize the radii of the screw and rod positions.
  void initializeScrewAndRodRadii();

  /// Initialize the screw type medium.
  void initializeScrewTypeMedium();

  /// Initialize the rod type medium.
  void initializeRodTypeMedium();

  /// Add a screw property set to the collection of total screws.
  /// \param  screwTypeID The screw type ID.
  /// \param  iRing       The ring number.
  /// \param  phi         Azimuthal angle of the screw location.
  void addScrewProperties(int screwTypeID, int iRing, float phi);

  /// Add a rod property set to the collection of total rods.
  /// \param  rodTypeID The rod type ID.
  /// \param  iRing     The ring number.
  void addRodProperties(int rodTypeID, int iRing);

  /// Initialize the position and dimension for every screw and rod.
  void initializeScrewAndRodPositionsAndDimensions();

  /// Initialize the sensitive volumes.
  void initializeSensVols();

  /// Initialize the non-sensitive volumes.
  void initializeNonSensVols();

  /// Initialize a composite shape of all screw holes. This shape is removed
  /// from all volumes that the screws are passing through to avoid overlaps.
  void initializeScrewHoles();

  /// Initialize a composite shape of all rod holes. This shape is removed
  /// from all volumes that the rods are passing through to avoid overlaps.
  void initializeRodHoles();

  /// Initialize cell volumes with a specified thickness and medium.
  /// \param  cellType    The type of the cells.
  /// \param  zThicknes   The thickness of the cells.
  /// \param  medium      The medium of the cells.
  /// \param  isSensitive Specifies if the cells are sensitive volumes.
  void initializeCells(const std::string& cellType, const float zThickness, const TGeoMedium* medium, bool isSensitive);

  /// Initialize scintillator cell volumes.
  void initializeScintCells();

  /// Initialize plastic cell volumes for optical fiber support.
  void initializePlasticCells();

  /// Initialize volumes equivalent to the optical fibers.
  void initializeFibers();

  /// Initialize the screw volumes.
  void initializeScrews();

  /// Initialize the rod volumes.
  void initializeRods();

  /// Initialize the metal container volume.
  void initializeMetalContainer();

  /// Assemble the sensitive volumes.
  /// \param  vFV0  The FIT V0 volume.
  void assembleSensVols(TGeoVolume* vFV0) const;

  /// Assemble the nonsensitive volumes.
  /// \param  vFV0  The FIT V0 volume.
  void assembleNonSensVols(TGeoVolume* vFV0) const;

  /// Assemble the scintillator sectors.
  /// \param  vFV0  The FIT V0 volume.
  void assembleScintSectors(TGeoVolume* vFV0) const;

  /// Assemble the plastice sectors.
  /// \param  vFV0  The FIT V0 volume.
  void assemblePlasticSectors(TGeoVolume* vFV0) const;

  /// Assemble the optical fibers.
  /// \param  vFV0  The FIT V0 volume.
  void assembleFibers(TGeoVolume* vFV0) const;

  /// Assemble the screwss.
  /// \param  vFV0  The FIT V0 volume.
  void assembleScrews(TGeoVolume* vFV0) const;

  /// Assemble the rods.
  /// \param  vFV0  The FIT V0 volume.
  void assembleRods(TGeoVolume* vFV0) const;

  /// Assemble the metal container.
  /// \param  vFV0  The FIT V0 volume.
  void assembleMetalContainer(TGeoVolume* vFV0) const;

  /// Build sector assembly of specified type.
  /// \param  cellName  The type of the cells in the sector assembly.
  /// \return The sector assembly.
  TGeoVolumeAssembly* buildSectorAssembly(const std::string& cellName) const;

  /// Build a sector of specified type and number.
  /// \param  cellType  The type of the cells in the sector.
  /// \param  iSector   The numbering of the sector.
  /// \return The sector.
  TGeoVolumeAssembly* buildSector(const std::string& cellType, int iSector) const;

  /// Create the shape for a specified screw.
  /// \param  shapeName   The name of the shape.
  /// \param  screwTypeID The number of the screw type.
  /// \param  xEpsilon    Shrinks or expands the x dimensions of the screw shape.
  /// \param  yEpsilon    Shrinks or expands the y dimensions of the screw shape.
  /// \param  zEpsilon    Shrinks or expands the z dimensions of the screw shape.
  /// \return The screw shape.
  TGeoShape* createScrewShape(const std::string& shapeName, int screwTypeID, float xEpsilon = 0, float yEpsilon = 0,
                              float zEpsilon = 0) const;

  /// Create the shape for a specified rod.
  /// \param  shapeName The name of the shape.
  /// \param  rodTypeID The number of the rod type.
  /// \param  xEpsilon  Shrinks or expands the x dimensions of the rod shape.
  /// \param  yEpsilon  Shrinks or expands the y dimensions of the rod shape.
  /// \param  zEpsilon  Shrinks or expands the z dimensions of the rod shape.
  /// \return The rod shape.
  TGeoShape* createRodShape(const std::string& shapeName, int rodTypeID, float xEpsilon = 0, float yEpsilon = 0,
                            float zEpsilon = 0) const;

  /// Helper function for creating and registering a TGeoTranslation.
  /// \param  name  The name of the translation.
  /// \param  dx    Translation dx.
  /// \param  dy    Translation dy.
  /// \param  dz    Translation dz.
  /// \return The newly created and registered TGeoTranslation.
  TGeoTranslation* createAndRegisterTrans(const std::string& name, double dx = 0, double dy = 0, double dz = 0) const;

  /// Helper function for creating and registering a TGeoRotation.
  /// \param  name  The name of the rotation.
  /// \param  dx    Translation phi.
  /// \param  dy    Translation theta.
  /// \param  dz    Translation psi.
  /// \return The newly created and registered TGeoRotation.
  TGeoRotation* createAndRegisterRot(const std::string& name, double phi = 0, double theta = 0, double psi = 0) const;

  /// Helper function for creating volume names.
  /// \param  volumeType  A string that will be included in the volume name.
  /// \param  number      A number, e.g. an ID, that is included in the name. A negative number is omitted.
  /// \return The volume name.
  const std::string createVolumeName(const std::string& volumeType, int number = -1) const;

  std::vector<std::string> mSensitiveVolumeNames; ///< The names of all the sensitive volumes

  /// Average ring radii
  ///
  /// index 0 -> ring 1 min, index 1 -> ring 1 max and ring 2 min, ... index 5 -> ring 5 max
  std::vector<float> mRAvgRing;
  std::vector<float> mRMinScintillator; ///< Inner radii of scintillator rings (.at(0) -> ring 1, .at(4) -> ring 5)
  std::vector<float> mRMaxScintillator; ///< Outer radii of scintillator rings (.at(0) -> ring 1, .at(4) -> ring 5)
  std::vector<float> mRMinFiber;        ///< Inner radii of fiber volumes (.at(0) -> fiber 1)
  std::vector<float> mRMaxFiber;        ///< Outer radii of fiber volumes (.at(0) -> fiber 1)

  /// Medium of the fiber volumes
  /// .at(n) -> medium of the n:th fiber starting from the middle.
  std::vector<TGeoMedium*> mMediumFiber;

  std::vector<float> mRScrewAndRod; ///< Radii of the screw and rod positions

  std::vector<float> mDrMinScrews; ///< Radii of the thinner part of the screws
  std::vector<float> mDrMaxScrews; ///< Radii of the thicker part of the screws
  std::vector<float> mDzMaxScrews; ///< Length of the thinner part of the screws
  std::vector<float> mDzMinScrews; ///< Length of the thicker part of the screws

  std::vector<float> mRScrews;    ///< Radial distance to the screw locations
  std::vector<int> mScrewTypeIDs; ///< The type ID of each screw (.at(n) -> type ID of screw no. n)

  std::vector<float> mDxMinRods; ///< Width of the thinner part of the rods
  std::vector<float> mDxMaxRods; ///< Width of the thicker part of the rods
  std::vector<float> mDyMinRods; ///< Height of the thinner part of the rods
  std::vector<float> mDyMaxRods; ///< Height of the thicker part of the rods
  std::vector<float> mDzMaxRods; ///< Length of the thinner part of the rods
  std::vector<float> mDzMinRods; ///< Length of the thicker part of the rods
  std::vector<int> mRodTypeIDs;  ///< The type ID of each rod (.at(n) -> type ID of rod no. n)

  std::vector<TGeoMatrix*> mSectorTrans;      ///< Transformations of sectors (.at(0) -> sector 1)
  std::vector<std::vector<float>> mScrewPos;  ///< xyz-coordinates of all the screws
  std::vector<std::vector<float>> mRodPos;    ///< xyz-coordinates of all the rods
  std::vector<TGeoMedium*> mMediumScrewTypes; ///< Medium of the screw types
  std::vector<TGeoMedium*> mMediumRodTypes;   ///< Medium of the rod types

  const int mGeometryType;                          ///< The type of the geometry.
  std::map<EGeoComponent, bool> mEnabledComponents; ///< Map of the enabled state of all geometry components
  TGeoMatrix* mLeftTransformation;                  ///< Transformation for the left part of the detector
  TGeoMatrix* mRightTransformation;                 ///< Transformation for the right part of the detector

  ClassDefNV(Geometry, 1);
};
} // namespace fv0
} // namespace o2
#endif
