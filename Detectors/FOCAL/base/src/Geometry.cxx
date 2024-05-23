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

// --- Standard library ---
#include <fstream>
#include <algorithm>

// --- ROOT system ---

#include <fairlogger/Logger.h>

#include "FOCALBase/Geometry.h"

using namespace o2::focal;

bool Geometry::sInit = false;
Geometry* Geometry::sGeom = nullptr;

//_________________________________________________________________________
Geometry::Geometry(Geometry* geo)
{
  std::fill_n(mPixelLayerLocations.begin(), 20, -1);
  std::fill_n(mSegments.begin(), 100, -100);
  std::fill_n(mNumberOfLayersInSegments.begin(), 100, -1);
  std::fill_n(mLocalLayerZ.begin(), 100, 0.0);
  std::fill_n(mLocalSegmentsZ.begin(), 100, 0.0);
  std::fill_n(mLayerThickness.begin(), 100, 0.0);

  *this = geo;
}

//_________________________________________________________________________
Geometry* Geometry::getInstance()
{

  if (sGeom == nullptr) {
    sGeom = new Geometry();
    sGeom->init();
  } else {
    if (sInit == false) {
      sGeom = new Geometry();
      sGeom->init();
    }
  }

  return sGeom;
}

//_________________________________________________________________________
Geometry* Geometry::getInstance(std::string filename)
{

  if (sGeom == nullptr) {
    sGeom = new Geometry();
    sGeom->init(filename);
  } else {
    if (sInit == false) {
      sGeom = new Geometry();
      sGeom->init(filename);
    }
  }
  return sGeom;
}

//_________________________________________________________________________
void Geometry::init(std::string filename)
{
  if (filename != "default") {
    setParameters(filename);
  } else {
    setParameters();
  }

  buildComposition();
  sInit = true;
}

//_________________________________________________________________________
void Geometry::init()
{
  setParameters();
  buildComposition();
  sInit = true;
}

//_________________________________________________________________________
void Geometry::buildComposition()
{
  mGeometryComposition.reserve(1000);

  int nlayers = mNPadLayers + mNPixelLayers + mNHCalLayers;

  ////// Si pad micro module  for the first Pad layer
  for (int i = 0; i < nlayers; i++) {
    if (i < mNPadLayers + mNPixelLayers) {
      // Check whether it is a pixel layer
      int isPixel = 0;
      for (int iPix = 0; iPix < mNPixelLayers && !isPixel; iPix++) {
        LOG(debug) << "Check pixel layer idx " << iPix << " loc " << mPixelLayerLocations[iPix] << " i= " << i;
        if (i == mPixelLayerLocations[iPix]) {
          isPixel = 1;
        }
      }

      if (isPixel) {
        // create pixel compositions
        LOG(debug) << "Adding pixel layer at layer " << i;
        for (auto& icomp : mPixelCompositionBase) {
          icomp.setLayerNumber(i);
          mGeometryComposition.push_back(icomp);
        }
        mLayerThickness[i] = mPixelLayerThickness;
      } else {
        // create pad compositions
        for (auto& icomp : mPadCompositionBase) {
          icomp.setLayerNumber(i);
          mGeometryComposition.push_back(icomp);
        }
        mLayerThickness[i] = mPadLayerThickness;
      }
    } else {
      // create hcal compositions
      for (auto& icomp : mHCalCompositionBase) {
        icomp.setLayerNumber(i);
        mGeometryComposition.push_back(icomp);
      }
      mLayerThickness[i] = mHCalLayerThickness;
    }
  } // end loop over nlayers

  for (int i = 0; i < nlayers; i++) {
    mLocalLayerZ[i] = 0;
    for (int j = 0; j < i; j++) {
      mLocalLayerZ[i] += mLayerThickness[j];
    }
  }
  for (int i = 0; i < nlayers; i++) {
    mLocalSegmentsZ[mSegments[i]] += mLayerThickness[i];
  }

  mLocalLayerZ[nlayers] = -1;

  ////// add the front matter to the tower
  for (auto& icomp : mFrontMatterCompositionBase) {
    icomp.setLayerNumber(-1);
    mGeometryComposition.push_back(icomp);
  }

  //// re-iterate to set the longitudinal positions
  for (auto& icomp : mGeometryComposition) {
    if (icomp.layer() >= 0) {
      icomp.setCenterZ(mFrontMatterLayerThickness + mLocalLayerZ[icomp.layer()] + icomp.centerZ() + icomp.sizeZ() / 2); /// this is pad/strip layer
    } else {
      icomp.setCenterZ(icomp.centerZ() + icomp.sizeZ() / 2); /// this is frontmatter
    }
  }
}

//_________________________________________________________________________
void Geometry::setParameters()
{

  LOG(warning) << " Default parameters are not used ";
  /////// this is default setting for the global parameters
  mGlobal_FOCAL_Z0 = 500;
}

//_________________________________________________________________________
void Geometry::setParameters(std::string geometryfile)
{
  /////// this is default setting for the global parameters
  mGlobal_FOCAL_Z0 = 700.0;
  mInsertFrontPadLayers = false;
  // PAD setup
  mGlobal_Pad_Size = 1.0;   // pad size
  mGlobal_PAD_NX = 9;       // number of X pads in wafer
  mGlobal_PAD_NY = 8;       // number of Y pads in wafer
  mGlobal_PAD_NX_Tower = 5; // number of X wafers in tower
  mGlobal_PAD_NY_Tower = 1; // number of Y wafers in tower
  mGlobal_PPTOL = 0.0;      // tolerance between the wafers
  mGlobal_PAD_SKIN = 0.2;   // dead area (guard ring) on the wafer
  mGlobal_PIX_SKIN = 0.004;
  // PIX setup
  mGlobal_Pixel_Readout = false;
  mGlobal_Pixel_Size = 0.005; // pixel size
  mGlobal_PIX_SizeX = 3.0;    // sensor size X
  mGlobal_PIX_SizeY = 2.74;   // sensor size Y
  mGlobal_PIX_OffsetX = 1.0;
  mGlobal_PIX_OffsetY = 0.09;
  mGlobal_PIX_NX_Tower = 15; // number of sensors in X
  mGlobal_PIX_NY_Tower = 3;  // number of sensors in Y

  mGlobal_Tower_NX = 2;
  mGlobal_Tower_NY = 11;

  mNPixelLayers = 4;
  mPixelLayerLocations[0] = 2;
  mPixelLayerLocations[1] = 4;
  mPixelLayerLocations[2] = 6;
  mPixelLayerLocations[3] = 8;

  mTowerSizeX = 0;
  mTowerSizeY = 0;
  mWaferSizeX = 0;
  mWaferSizeY = 0;

  std::ifstream fin(geometryfile);
  if (fin.fail()) {
    LOG(error) << "No geometry file for FoCAL. Use default ones";
    setParameters();
    return;
  } else {
    LOG(info) << "Using geometry file " << geometryfile;
  }

  std::vector<Composition> padCompDummy(10);
  std::vector<Composition> hCalCompDummy(10);
  std::vector<Composition> pixelCompDummy(10);
  std::vector<Composition> frontMatterCompDummy(10);
  int nPad = 0;
  int hHCal = 0;
  int nPixel = 0;
  int nFrontMatter = 0;

  int npadlayers = 0;
  int npixlayers = 0;
  int pixl[10];

  std::string input;

  LOG(info) << "Loading FOCAL geometry file ";
  while (getline(fin, input)) {
    LOG(debug) << "Read string :: " << input.c_str();
    const char* p = std::strpbrk("#", input.c_str());
    if (p != nullptr) {
      LOG(debug) << "Skipping comment";
      continue;
    }

    std::vector<std::string> tokens;
    std::stringstream str(input);
    std::string tmpStr;
    while (getline(str, tmpStr, ' ')) {
      if (tmpStr.empty()) {
        continue;
      }
      tokens.push_back(tmpStr);
    }

    std::string command = tokens[0];
    LOG(debug) << "command: " << command;

    if (command.find("COMPOSITION") != std::string::npos) /// definition of the composition
    {
      std::string material = tokens[1];
      float cx = std::stof(tokens[2]);
      float cy = std::stof(tokens[3]);
      float dx = std::stof(tokens[4]);
      float dy = std::stof(tokens[5]);
      float dz = std::stof(tokens[6]);
      float cz = 0;

      LOG(debug) << "Material :: " << material;
      LOG(debug) << "cx/cy/dx/dy/dz :: " << cx << " / " << cy << " / " << dx << " / " << dy << " / " << dz;

      int stack;
      if (command.find("PAD") != std::string::npos) {
        sscanf(command.c_str(), "COMPOSITION_PAD_S%d", &stack);
        padCompDummy.emplace_back(material, stack, stack, 0, cx, cy, cz, dx, dy, dz);
        nPad++;
      }

      if (command.find("HCAL") != std::string::npos) {
        sscanf(command.c_str(), "COMPOSITION_HCAL_S%d", &stack);
        hCalCompDummy.emplace_back(material, stack, stack, 0, cx, cy, cz, dx, dy, dz);
        hHCal++;
      }

      if (command.find("PIX") != std::string::npos) {
        sscanf(command.c_str(), "COMPOSITION_PIX_S%d", &stack);
        pixelCompDummy.emplace_back(material, stack, stack, 0, cx, cy, cz, dx, dy, dz);
        mGlobal_PIX_SizeX = dx;
        mGlobal_PIX_SizeY = dy;
        nPixel++;
      }

      if (command.find("FM") != std::string::npos) {
        sscanf(command.c_str(), "COMPOSITION_FM_S%d", &stack);
        frontMatterCompDummy.emplace_back(material, stack, stack, 0, cx, cy, cz, dx, dy, dz);
        nFrontMatter++;
      }
    } // end if COMPOSITION

    if (command.find("GLOBAL") != std::string::npos) {

      if (command.find("PAD_SIZE") != std::string::npos) {
        mGlobal_Pad_Size = std::stof(tokens[1]);
        LOG(debug) << "Pad sensor size is set to : " << mGlobal_Pad_Size;
      }

      if (command.find("PAD_NX") != std::string::npos) {
        mGlobal_PAD_NX = std::stoi(tokens[1]);
        LOG(debug) << "No. sensors per pad wafer in X direction : " << mGlobal_PAD_NX;
      }

      if (command.find("PAD_NY") != std::string::npos) {
        mGlobal_PAD_NY = std::stoi(tokens[1]);
        LOG(debug) << "No. sensors per pad wafer in Y direction : " << mGlobal_PAD_NY;
      }

      if (command.find("PAD_SUPERMODULE_X") != std::string::npos) {
        mGlobal_PAD_NX_Tower = std::stoi(tokens[1]);
        LOG(debug) << "Number of pad wafers per module in X direction : " << mGlobal_PAD_NX_Tower;
      }

      if (command.find("PAD_SUPERMODULE_Y") != std::string::npos) {
        mGlobal_PAD_NY_Tower = std::stoi(tokens[1]);
        LOG(debug) << "Number of pad wafers per module in Y direction : " << mGlobal_PAD_NY_Tower;
      }

      if (command.find("PIX_NX") != std::string::npos) {
        mGlobal_PIX_NX_Tower = std::stoi(tokens[1]);
        LOG(debug) << "Number of pixels sensors per module in X direction : " << mGlobal_PIX_NX_Tower;
      }

      if (command.find("PIX_NY") != std::string::npos) {
        mGlobal_PIX_NY_Tower = std::stoi(tokens[1]);
        LOG(debug) << "Number of pixels sensors per module in Y direction : " << mGlobal_PIX_NY_Tower;
      }

      if (command.find("PAD_PPTOL") != std::string::npos) {
        mGlobal_PPTOL = std::stof(tokens[1]);
        LOG(debug) << "Distance between pad sensors : " << mGlobal_PPTOL;
      }

      if (command.find("PAD_SKIN") != std::string::npos) {
        mGlobal_PAD_SKIN = std::stof(tokens[1]);
        LOG(debug) << "Pad wafer skin : " << mGlobal_PAD_SKIN;
      }

      if (command.find("FOCAL_Z") != std::string::npos) {
        mGlobal_FOCAL_Z0 = std::stof(tokens[1]);
        LOG(debug) << "Z-Location of the FoCAL is set to : " << mGlobal_FOCAL_Z0;
      }

      if (command.find("HCAL_TOWER_SIZE") != std::string::npos) {
        mGlobal_HCAL_Tower_Size = std::stof(tokens[1]);
        LOG(debug) << "The size of the HCAL readout tower will be : " << mGlobal_HCAL_Tower_Size;
      }

      if (command.find("HCAL_TOWER_NX") != std::string::npos) {
        mGlobal_HCAL_Tower_NX = std::stoi(tokens[1]);
        LOG(debug) << "The number of the HCAL readout towers in X will be : " << mGlobal_HCAL_Tower_NX;
      }

      if (command.find("HCAL_TOWER_NY") != std::string::npos) {
        mGlobal_HCAL_Tower_NY = std::stoi(tokens[1]);
        LOG(debug) << "The number of the HCAL readout towers in Y will be : " << mGlobal_HCAL_Tower_NY;
      }

      if (command.find("PIX_OffsetX") != std::string::npos) {
        mGlobal_PIX_OffsetX = std::stof(tokens[1]);
        LOG(debug) << "Pixel offset from the beam pipe will be: " << mGlobal_PIX_OffsetX;
      }

      if (command.find("PIX_OffsetY") != std::string::npos) {
        mGlobal_PIX_OffsetY = std::stof(tokens[1]);
        LOG(debug) << "Pixel offset from the top of module will be: " << mGlobal_PIX_OffsetY;
      }

      if (command.find("PIX_SKIN") != std::string::npos) {
        mGlobal_PIX_SKIN = std::stof(tokens[1]);
        LOG(debug) << "Pixel sensor skin : " << mGlobal_PIX_SKIN;
      }

      if (command.find("TOWER_TOLX") != std::string::npos) {
        mGlobal_TOWER_TOLX = std::stof(tokens[1]);
        mGlobal_Gap_Material = tokens[2];
        LOG(debug) << "Distance between modules in X direction : " << mGlobal_TOWER_TOLX << ", Material : " << mGlobal_Gap_Material;
      }

      if (command.find("TOWER_TOLY") != std::string::npos) {
        mGlobal_TOWER_TOLY = std::stof(tokens[1]);
        mGlobal_Gap_Material = tokens[2];
        LOG(debug) << "Distance between modules in Y direction : " << mGlobal_TOWER_TOLY << " Material : " << mGlobal_Gap_Material;
      }

      if (command.find("MIDDLE_TOWER_OFFSET") != std::string::npos) {
        mGlobal_Middle_Tower_Offset = std::stof(tokens[1]);
        LOG(debug) << "Middle tower offset will be: " << mGlobal_Middle_Tower_Offset;
      }

      if (command.find("Tower_NX") != std::string::npos) {
        mGlobal_Tower_NX = std::stof(tokens[1]);
        LOG(debug) << "Number of FOCAL modules in x direction is set to : " << mGlobal_Tower_NX;
      }

      if (command.find("Tower_NY") != std::string::npos) {
        mGlobal_Tower_NY = std::stof(tokens[1]);
        LOG(debug) << "Number of FOCAL modules in y direction is set to : " << mGlobal_Tower_NY;
      }
    } // end if GLOBAL

    if (command.find("COMMAND") != std::string::npos) {

      if (command.find("NUMBER_OF_PAD_LAYERS") != std::string::npos) {
        npadlayers = std::stoi(tokens[1]);
        LOG(debug) << "Number of pad layers " << npadlayers;
      }

      if (command.find("NUMBER_OF_HCAL_LAYERS") != std::string::npos) {
        mNHCalLayers = std::stoi(tokens[1]);
        LOG(debug) << "Number of HCAL layers " << mNHCalLayers;
        if (mNHCalLayers == 1) {
          mUseSandwichHCAL = false;
        } else {
          mUseSandwichHCAL = true;
        }
      }

      if (command.find("NUMBER_OF_SEGMENTS") != std::string::npos) {
        mNumberOfSegments = std::stoi(tokens[1]);
        LOG(debug) << "Number of Segments " << mNumberOfSegments;
      }

      if (command.find("INSERT_PIX") != std::string::npos) {
        sscanf(command.c_str(), "COMMAND_INSERT_PIX_AT_L%d", &pixl[npixlayers]);
        LOG(debug) << "Number of pixel layer " << npixlayers << " : location " << pixl[npixlayers];
        npixlayers++;
      }

      if (command.find("COMMAND_PIXEL_READOUT_ON") != std::string::npos) {
        mGlobal_Pixel_Readout = true;
        mGlobal_Pixel_Size = std::stof(tokens[1]);
        LOG(debug) << "Pixel readout on (for MASPS): pixel size is set to : " << mGlobal_Pixel_Size;
      }

      if (command.find("COMMAND_INSERT_FRONT_PAD_LAYERS") != std::string::npos) {
        mInsertFrontPadLayers = true;
        LOG(debug) << "Insert two pad layers in front of ECAL for charged particle veto!";
      }

      if (command.find("COMMAND_INSERT_HCAL_READOUT") != std::string::npos) {
        mInsertFrontHCalReadoutMaterial = true;
        LOG(debug) << "Insert Aluminium 1cm thick layer behind HCAL to simulate readout SiPM material !";
      }
    } // end if COMMAND

    if (command.find("VIRTUAL") != std::string::npos) {

      int segment, minlayer, maxLayer, isPixel;
      float padSize, sensitiveThickness, pixelTreshold;

      if (command.find("N_SEGMENTS") != std::string::npos) {
        mVirtualNSegments = std::stoi(tokens[1]);
        LOG(debug) << "Number of Virtual Segments is set to : " << mVirtualNSegments;
      }

      if (command.find("SEGMENT_LAYOUT") != std::string::npos) {
        minlayer = std::stoi(tokens[1]);
        maxLayer = std::stoi(tokens[2]);
        padSize = std::stof(tokens[3]);
        sensitiveThickness = std::stof(tokens[4]);
        isPixel = std::stoi(tokens[5]);
        pixelTreshold = std::stof(tokens[6]);

        if (mVirtualSegmentComposition.size() == 0) {
          if (mVirtualNSegments <= 0) {
            LOG(debug) << "Making 20 segments";
            for (int seg = 0; seg < 20; seg++) {
              mVirtualSegmentComposition.emplace_back();
            }
            mVirtualNSegments = 20;
          } else {
            LOG(debug) << "Making " << mVirtualNSegments << " segments";
            for (int seg = 0; seg < mVirtualNSegments; seg++) {
              mVirtualSegmentComposition.emplace_back();
            }
          }
        }

        sscanf(command.c_str(), "VIRTUAL_SEGMENT_LAYOUT_N%d", &segment);
        if (segment > mVirtualNSegments) {
          continue;
        }
        mVirtualSegmentComposition[segment].mMinLayer = minlayer;
        mVirtualSegmentComposition[segment].mMaxLayer = maxLayer;
        mVirtualSegmentComposition[segment].mPadSize = padSize;
        mVirtualSegmentComposition[segment].mRelativeSensitiveThickness = sensitiveThickness;
        mVirtualSegmentComposition[segment].mPixelTreshold = pixelTreshold;
        mVirtualSegmentComposition[segment].mIsPixel = isPixel;

        LOG(debug) << "Segment number " << segment << " defined with (minLayer, maxLayer, padSize, isPixel): ("
                   << minlayer << ", " << maxLayer << ", " << padSize << ", " << isPixel << ")";
      } // end if SEGMENT_LAYOUT

    } // end if VIRTUAL

  } // end while

  setUpTowerWaferSize();
  /////// re-arrange the longitudinal components
  mNPixelLayers = npixlayers;
  for (int i = 0; i < npixlayers; i++) {
    mPixelLayerLocations[i] = pixl[i];
  }

  mNPadLayers = npadlayers;
  LOG(debug) << "mNPadLayers, mNPixelLayers, mNHCalLayers, mNumberOfSegments :: " << mNPadLayers << " / " << mNPixelLayers << " / " << mNHCalLayers << " / " << mNumberOfSegments;

  mLayerSeg = (mNPadLayers + mNPixelLayers + mNHCalLayers) / mNumberOfSegments;

  if (mNumberOfSegments >= 100) {
    LOG(warning) << "You reached the segments limits! Setting Number of segments to: 100";
    mNumberOfSegments = 99;
    LOG(warning) << "New number of segments: " << mNumberOfSegments;
    mLayerSeg = (mNPadLayers + mNPixelLayers + mNHCalLayers) / mNumberOfSegments;
  }
  if ((mNPadLayers + mNPixelLayers + mNHCalLayers) % mNumberOfSegments) {
    mNumberOfSegments++;
    for (int i = 0; i < mNumberOfSegments; i++) {
      mNumberOfLayersInSegments[i] = mLayerSeg;
    }
    LOG(debug) << "Number of segments: " << mNumberOfSegments;
  } else {
    for (int i = 0; i < mNumberOfSegments; i++) {
      mNumberOfLayersInSegments[i] = mLayerSeg;
    }
  }

  setUpLayerSegmentMap();

  float center_z = 0;

  mPadCompositionBase.reserve(200);
  mHCalCompositionBase.reserve(200);
  mPixelCompositionBase.reserve(200);
  mFrontMatterCompositionBase.reserve(200);

  for (auto& tmpComp : padCompDummy) {
    LOG(debug) << "Material(pad layer) " << tmpComp.material();
    LOG(debug) << "layer / stack / id :: " << tmpComp.layer() << " / " << tmpComp.stack() << " / " << tmpComp.id();
    LOG(debug) << "center x,y,dz :: " << tmpComp.sizeX() << " / " << tmpComp.sizeY() << " / " << tmpComp.sizeZ();
    if (tmpComp.material().compare("SiPad")) { // materials other than SiPad
      mPadCompositionBase.emplace_back(tmpComp.material(), tmpComp.layer(), tmpComp.stack(), tmpComp.id(),
                                       tmpComp.centerX(), tmpComp.centerY(), center_z,
                                       mTowerSizeX, mTowerSizeY, tmpComp.sizeZ());
      if (mTowerSizeX < tmpComp.sizeX()) {
        mTowerSizeX = tmpComp.sizeX();
      }
      if (mTowerSizeY < tmpComp.sizeY()) {
        mTowerSizeY = tmpComp.sizeY();
      }
    } else {
      for (int itowerX = 0; itowerX < mGlobal_PAD_NX_Tower; itowerX++) {
        for (int itowerY = 0; itowerY < mGlobal_PAD_NY_Tower; itowerY++) {
          for (int ix = 0; ix < mGlobal_PAD_NX; ix++) {
            for (int iy = 0; iy < mGlobal_PAD_NY; iy++) {
              auto [x, y] = getGeoPadCenterLocal(itowerX, itowerY, iy, ix);
              mPadCompositionBase.emplace_back("SiPad", tmpComp.layer(), tmpComp.stack(),
                                               ix + iy * mGlobal_PAD_NX + itowerX * mGlobal_PAD_NX * mGlobal_PAD_NY + itowerY * mGlobal_PAD_NX_Tower * mGlobal_PAD_NX * mGlobal_PAD_NY,
                                               x, y, center_z,
                                               mGlobal_Pad_Size, mGlobal_Pad_Size, tmpComp.sizeZ());
              if (mTowerSizeX < mGlobal_Pad_Size) {
                mTowerSizeX = mGlobal_Pad_Size;
              }
              if (mTowerSizeY < mGlobal_Pad_Size) {
                mTowerSizeY = mGlobal_Pad_Size;
              }
            }
          }
        } // end for itowerY
      }   // end for itowerX
    }     // end else
    center_z += tmpComp.getThickness();
  } // end loop over pad layer compositions
  LOG(debug) << "============ Created all pad layer compositions (" << mPadCompositionBase.size() << " volumes)";

  mPadLayerThickness = center_z;

  mGlobal_PIX_OffsetY = (getTowerSizeY() - mGlobal_PIX_NY_Tower * mGlobal_PIX_SizeY) / 2 - 2.0 * mGlobal_PIX_SKIN;

  center_z = 0;
  for (auto& tmpComp : pixelCompDummy) {
    LOG(debug) << "Material (pixel layer) " << tmpComp.material();
    LOG(debug) << "layer / stack / id :: " << tmpComp.layer() << " / " << tmpComp.stack() << " / " << tmpComp.id();
    LOG(debug) << "center x,y,dz :: " << tmpComp.sizeX() << " / " << tmpComp.sizeY() << " / " << tmpComp.sizeZ();
    if (tmpComp.material().compare("SiPix")) {
      mPixelCompositionBase.emplace_back(tmpComp.material(), mPixelLayerLocations[0], tmpComp.stack(), tmpComp.id(),
                                         tmpComp.centerX(), tmpComp.centerY(), center_z, mTowerSizeX, mTowerSizeY, tmpComp.sizeZ());
    } else {
      for (int ix = 0; ix < mGlobal_PIX_NX_Tower; ix++) {
        for (int iy = 0; iy < mGlobal_PIX_NY_Tower; iy++) {
          auto [pixX, pixY] = getGeoPixCenterLocal(iy, ix);
          mPixelCompositionBase.emplace_back("SiPix", tmpComp.layer(), tmpComp.stack(), ix + iy * mGlobal_PIX_NX_Tower,
                                             pixX, pixY, center_z,
                                             mGlobal_PIX_SizeX, mGlobal_PIX_SizeY, tmpComp.sizeZ());
        }
      }
    }
    center_z += tmpComp.getThickness();
  }
  LOG(debug) << "============ Created all pixel layer compositions (" << mPixelCompositionBase.size() << " volumes)";
  mPixelLayerThickness = center_z;

  // Add HCal Layers
  center_z = 0;
  for (auto& tmpComp : hCalCompDummy) {
    LOG(debug) << "Material (hcal) " << tmpComp.material();
    LOG(debug) << "layer / stack / id :: " << tmpComp.layer() << " / " << tmpComp.stack() << " / " << tmpComp.id();
    LOG(debug) << "center x,y,dz :: " << tmpComp.sizeX() << " / " << tmpComp.sizeY() << " / " << tmpComp.sizeZ();
    mHCalCompositionBase.emplace_back(tmpComp.material(), tmpComp.layer(), tmpComp.stack(), tmpComp.id(),
                                      tmpComp.centerX(), tmpComp.centerY(), mNHCalLayers == 1 ? 0. : center_z, // if we decided to use the spagetti HCAL it will be only one layer with two compositions
                                      tmpComp.sizeX(), tmpComp.sizeY(), tmpComp.sizeZ());
    if (mNHCalLayers == 1) {
      center_z = tmpComp.getThickness();
    } else {
      center_z += tmpComp.getThickness();
    }
  }
  LOG(debug) << "============ Created all hcal compositions (" << mHCalCompositionBase.size() << " volumes)";
  mHCalLayerThickness = center_z;
  center_z = 0;

  mFrontMatterLayerThickness = center_z;
  LOG(debug) << " end of SetParameters ";
}

//_________________________________________________________________________
const Composition* Geometry::getComposition(int layer, int stack) const
{

  for (auto& icomp : mGeometryComposition) {
    if (icomp.layer() == layer && icomp.stack() == stack) {
      return &icomp;
    }
  }
  return nullptr;
}

//_________________________________________________________________________
/// this gives global position of the center of tower
std::tuple<double, double, double> Geometry::getGeoTowerCenter(int tower, int segment) const
{
  int id = tower;
  int itowerx = id % getNumberOfTowersInX();
  int itowery = id / getNumberOfTowersInX();

  float dwx = getTowerSizeX() + getTowerGapSizeX();
  float dwy = getTowerSizeY() + getTowerGapSizeY();

  double x = itowerx * dwx + 0.5 * dwx - 0.5 * getFOCALSizeX();
  double y = itowery * dwy + 0.5 * dwy - 0.5 * getFOCALSizeY();
  if (itowerx == 0 && itowery == 5) {
    x -= mGlobal_Middle_Tower_Offset;
  }
  if (itowerx == 1 && itowery == 5) {
    x += mGlobal_Middle_Tower_Offset;
  }

  // From here is HCal stuff
  if (getVirtualIsHCal(segment)) {
    auto [status, nCols, nRows] = getVirtualNColRow(segment);
    int ix = id % nCols;
    int iy = id / nRows;

    if (mUseSandwichHCAL) {
      float padSize = mVirtualSegmentComposition[segment].mPadSize;
      double hCALsizeX = nCols * padSize;
      double hCALsizeY = nRows * padSize;
      x = ix * padSize + 0.5 * padSize - 0.5 * hCALsizeX;
      y = iy * padSize + 0.5 * padSize - 0.5 * hCALsizeY;
    } else {
      nCols = std::floor(getFOCALSizeX() / getHCALTowerSize() + 0.001) + 1;
      nRows = std::floor(getFOCALSizeY() / getHCALTowerSize() + 0.001);
      ix = id % nCols;
      iy = id / nRows;
      double beamPipeRadius = 3.6;                                 // in cm   TODO: check if this is OK
      double towerHalfDiag = std::sqrt(2) * 0.5 * getTowerSizeX(); // tower half diagonal
      double minRadius = beamPipeRadius + towerHalfDiag;

      float towerSize = getHCALTowerSize() / 7; // To be set from outside (number of channels on x & y)
      y = iy * towerSize + 0.5 * towerSize - 0.5 * towerSize * nRows;
      x = ix * towerSize + 0.5 * towerSize - 0.5 * towerSize * nCols;
      if (y < minRadius && y > -minRadius) {
        x = int(x) <= 0 ? x - (minRadius - towerSize) : x + (minRadius - towerSize);
      }
    }
  }

  /*
  //// remove beam pipe area
  //  define beam pipe radius, calculate half of the tower diagonal in XY
  //  and remove every tower which center is closer than the sum of the two...
  double beamPipeRadius = 3.6; // in cm        TODO: check if this is OK
  double towerHalfDiag = std::sqrt(2)*0.5*getTowerSizeX(); // tower half diagonal
  double minRadius = beamPipeRadius+towerHalfDiag;
  //
  if((x*x+y*y) < (minRadius*minRadius)){  // comparing the tower center position with the minimum distance in second powers.
    //mDisableTowers.push_back(Tower+1);
    //return false;
  }
  */

  return {x, y, getFOCALZ0()};
}

//_________________________________________________________________________
std::tuple<double, double, double> Geometry::getGeoCompositionCenter(int tower, int layer, int stack) const
{
  auto [status, segment] = getVirtualSegmentFromLayer(layer);
  auto [towX, towY, towZ] = getGeoTowerCenter(tower, segment);
  double z = towZ;

  Composition* comp1 = (Composition*)getComposition(layer, stack);
  if (comp1 == nullptr) {
    z = z + mLocalLayerZ[layer] - getFOCALSizeZ() / 2;
  } else {
    z = comp1->centerZ() - getFOCALSizeZ() / 2 + getFOCALZ0();
  }
  return {towX, towY, z};
}

//_________________________________________________________________________
/// this gives global position of the pad
std::tuple<double, double, double> Geometry::getGeoPadCenter(int tower, int layer, int stack, int row, int col) const
{
  auto [x, y, z] = getGeoCompositionCenter(tower, layer, stack);
  int itowerx = tower % mGlobal_PAD_NX_Tower;
  int itowery = tower / mGlobal_PAD_NX_Tower;
  auto [padX, padY] = getGeoPadCenterLocal(itowerx, itowery, row, col);

  return {x + padX, y + padY, z};
}

//_________________________________________________________________________
/// this gives local position of the pad with respect to the wafer
std::tuple<double, double> Geometry::getGeoPadCenterLocal(int towerX, int towerY, int row, int col) const
{
  /// startting to count from upper-left
  /*
     (0,0)
     ___________________
     |  __   __
     | |__| |__|
     |  __   __
     | |__| |__|
     |  __   __
     | |__| |__|
     |
   */
  double x = +towerX * mWaferSizeX + mGlobal_PAD_SKIN + col * (mGlobal_Pad_Size + mGlobal_PPTOL) + 0.5 * mGlobal_Pad_Size;
  double y = -towerY * mWaferSizeY - mGlobal_PAD_SKIN - row * (mGlobal_Pad_Size + mGlobal_PPTOL) - 0.5 * mGlobal_Pad_Size;
  x = x - 0.5 * getTowerSizeX();
  y = y + 0.5 * mTowerSizeY;
  return {x, y};
}

/// this gives local position of the pad with respect to the wafer
std::tuple<double, double> Geometry::getGeoPixCenterLocal(int row, int col) const
{
  /// startting to count from upper-left
  /*
     (0,0)
     ___________________
     |  __   __
     | |__| |__|
     |  __   __
     | |__| |__|
     |  __   __
     | |__| |__|
     |
   */
  double x = +col * (mGlobal_PIX_SizeX + 2.0 * mGlobal_PIX_SKIN) + 0.5 * mGlobal_PIX_SizeX;
  double y = -row * (mGlobal_PIX_SizeY + 2.0 * mGlobal_PIX_SKIN) - 0.5 * mGlobal_PIX_SizeY;
  x = x - 0.5 * mTowerSizeX;
  y = y + 0.5 * mTowerSizeY - mGlobal_PIX_OffsetY;
  return {x, y};
}

//_________________________________________________________________________
double Geometry::getTowerSizeX() const
{
  return mTowerSizeX;
  //  return mGlobal_NX_NY_Pads*(mGlobal_Pad_Size+mGlobal_PPTOL)-mGlobal_PPTOL+2*mGlobal_PAD_SKIN;
}

//_________________________________________________________________________
double Geometry::getTowerSizeY() const
{
  return mTowerSizeY;
  //  return mGlobal_NX_NY_Pads*(mGlobal_Pad_Size+mGlobal_PPTOL)-mGlobal_PPTOL+2*mGlobal_PAD_SKIN;
}

//_________________________________________________________________________
double Geometry::getFOCALSizeX() const
{
  return mGlobal_Tower_NX * (getTowerSizeX() + mGlobal_TOWER_TOLX);
}

//_________________________________________________________________________
double Geometry::getFOCALSizeY() const
{
  return mGlobal_Tower_NY * (getTowerSizeY() + mGlobal_TOWER_TOLY);
}

//_________________________________________________________________________
double Geometry::getFOCALSizeZ() const
{

  double ret = 0;
  for (int i = 0; i < mNPadLayers + mNPixelLayers + mNHCalLayers; i++) {
    ret += mLayerThickness[i];
  }
  ret = ret + mFrontMatterLayerThickness;
  return ret;
}

//_________________________________________________________________________
double Geometry::getECALSizeZ() const
{

  double ret = 0;
  for (int i = 0; i < mNPadLayers + mNPixelLayers; i++) {
    ret += mLayerThickness[i];
  }
  ret = ret + mFrontMatterLayerThickness;
  return ret;
}

//_________________________________________________________________________
double Geometry::getECALCenterZ() const
{

  // Determines the ECAL z center of mass with respect to the FOCAL
  double centerZ = mFrontMatterLayerThickness + mLocalLayerZ[0] + getECALSizeZ() / 2;
  return centerZ;
}

//_________________________________________________________________________
double Geometry::getHCALSizeZ() const
{

  double ret = 0;
  for (int i = mNPadLayers + mNPixelLayers; i < mNPadLayers + mNPixelLayers + mNHCalLayers; i++) {
    ret += mLayerThickness[i];
  }
  return ret;
}

//_________________________________________________________________________
double Geometry::getHCALCenterZ() const
{

  double centerZ = mFrontMatterLayerThickness + mLocalLayerZ[mNPadLayers + mNPixelLayers] + getHCALSizeZ() / 2;
  return centerZ;
}

//_________________________________________________________________________
// this returns the following quantities for the pad position location
// layer depth
// pad row and col in the wafer
// wafer id in the brick, where the pad belongs to
std::tuple<int, int, int, int, int, int, int> Geometry::getPadPositionId2RowColStackLayer(int id) const
{

  ////  id contains loction of pads in the tower, pad stack, pad layer
  /////  (fComp->id()) + (fComp->stack() << 12) + (fComp->layer() << 16) +1 ;
  /////
  int number = id - 1;
  int padid = (number & 0xfff);
  int stack = (number >> 12) & 0x000f;
  // lay = (number >> 16) & 0x00ff;
  int lay = (number >> 16) & 0x000f;

  // seg = fSegments[lay];
  auto [status, seg] = getVirtualSegmentFromLayer(lay); // NOTE: to be checked since this overrides the initialization above
  /*col = padid%mGlobal_PAD_NX;
  row = padid/mGlobal_PAD_NX;*/
  int waferx = 0;
  int wafery = 0;
  int col = 0;
  int row = 0;

  // This gives the (col,row) of the pixel sensor
  if (getVirtualIsPixel(seg)) {
    col = padid % mGlobal_PIX_NX_Tower;
    row = padid / mGlobal_PIX_NX_Tower;
  } else {
    col = padid % mGlobal_PAD_NX;
    int remainder = (padid - col) / mGlobal_PAD_NX;
    row = remainder % mGlobal_PAD_NY;
    remainder = (remainder - row) / mGlobal_PAD_NY;
    waferx = remainder % mGlobal_PAD_NX_Tower;
    wafery = remainder / mGlobal_PAD_NX_Tower;
  }
  /*cout << "FROM GEOMETRY  stack/lay/seg/waferx/wafery/col/row :: " << stack << " / " << lay << " / " << seg << " / "
       << waferx << " / " << wafery << " / " << col << " / " << row << endl;*/
  if (getVirtualIsHCal(seg)) {
    auto [status, nCols, nRows] = getVirtualNColRow(seg);
    col = id % nCols;
    row = id / nRows;
  }

  return {row, col, stack, lay, seg, waferx, wafery};
}

//_________________________________________________________________________
//// this gives longitudinal position of the segment
double Geometry::getFOCALSegmentZ(int seg) const
{

  double ret = 0;
  if (seg < 0 || seg > mNumberOfSegments) {
    ret = getFOCALZ0();
  } else {
    for (int i = 0; i < seg; i++) {
      ret += mLocalSegmentsZ[i];
    }
  }
  ret = ret + mLocalSegmentsZ[seg] / 2 + getFOCALZ0() - getFOCALSizeZ() / 2;
  return ret;
}

//_________________________________________________________________________
/// this function defines:
/// layer is pixel or pad
/// which segment this layer belongs to?
void Geometry::setUpLayerSegmentMap()
{
  ///// define the longitudinal elements
  ////  mSegments = -1 -> strip layer
  ////  mSegments = 0  --> pad 0th segement
  ////  mSegments = 1  --> pad 1th segement
  ////  mSegments = 2  --> pad 2th segement

  std::vector<int> layerType;
  for (int j = 0; j < mNPixelLayers + mNPadLayers + mNHCalLayers; j++) {
    layerType.push_back(0);
  }
  for (int i = 0; i < mNPixelLayers; i++) {
    layerType[mPixelLayerLocations[i]] = -1;
  }

  int low = 0;
  int start = 0;
  int high = 0;
  for (int i = 0; i < mNumberOfSegments; i++) {
    high += mNumberOfLayersInSegments[i];
    for (int j = start; j < mNPixelLayers + mNPadLayers + mNHCalLayers; j++) {
      if (layerType[j] == -1) {
        mSegments[j] = i;
        start++;
      } else {
        mSegments[j] = i;
        low++;
        start++;
      }
      if (low >= high) {
        break;
      }
    }
  }
}

//_________________________________________________________________________
/// this is the pixel number to be stored in the Hits.root file
/// this is used for the study with fine pixel readout
/// the pad is divided into the pixels with the size of mGlobal_ Pixel_Size
int Geometry::getPixelNumber(int vol0, int vol1, int /*vol2*/, double x, double y) const
{
  int ret = 0;
  if (mGlobal_Pixel_Readout == false) {
    ret = -1;
    return ret;
  }
  int id = vol0;
  // int tower = vol1;
  // int brick = vol2;  /// meaning 0 in the current design

  auto [row, col, stack, layer, segment, waferX, waferY] = getPadPositionId2RowColStackLayer(id);
  auto [pixX, pixY] = getGeoPixCenterLocal(row, col);

  double x_loc = x - pixX;
  double y_loc = y - pixY;
  double pixel_nbr_x = ((x_loc + 0.5 * getGlobalPixelWaferSizeX()) / (mGlobal_Pixel_Size));
  double pixel_nbr_y = ((y_loc + 0.5 * getGlobalPixelWaferSizeY()) / (mGlobal_Pixel_Size));

  int pixel_number_x;
  pixel_number_x = static_cast<int>(pixel_nbr_x);
  //  if(pixel_number_x-pixel_nbr_x>0.5){
  //    pixel_number_x = pixel_number_x+1;
  //  }

  int pixel_number_y;
  pixel_number_y = static_cast<int>(pixel_nbr_y);
  //  if(pixel_number_y-pixel_nbr_y>0.5){
  //    pixel_number_y = pixel_number_y+1;
  //  }
  ret = (pixel_number_x << 16) | pixel_number_y;
  // cout<<x<<" "<<y<<" "<<x0<<" "<<y0<<" "<<x_loc<<" "<<y_loc<<" "<<pixel_number_x<<" "<<pixel_number_y<<" "<<ret<<endl;
  return ret;
}

//_________________________________________________________________________
void Geometry::setUpTowerWaferSize()
{

  mWaferSizeX = mGlobal_PAD_NX * (mGlobal_Pad_Size + mGlobal_PPTOL) - mGlobal_PPTOL + 2 * mGlobal_PAD_SKIN;
  mWaferSizeY = mGlobal_PAD_NY * (mGlobal_Pad_Size + mGlobal_PPTOL) - mGlobal_PPTOL + 2 * mGlobal_PAD_SKIN;

  if (mTowerSizeX < mWaferSizeX * mGlobal_PAD_NX_Tower) {
    mTowerSizeX = mWaferSizeX * mGlobal_PAD_NX_Tower;
  }
  if (mTowerSizeY < mWaferSizeY * mGlobal_PAD_NY_Tower) {
    mTowerSizeY = mWaferSizeY * mGlobal_PAD_NY_Tower;
  }
  if (mTowerSizeX < mGlobal_PIX_SizeX * mGlobal_PIX_NX_Tower) {
    mTowerSizeX = mGlobal_PIX_SizeX * mGlobal_PIX_NX_Tower;
  }
  if (mTowerSizeY < mGlobal_PIX_SizeY * mGlobal_PIX_NY_Tower) {
    mTowerSizeY = mGlobal_PIX_SizeY * mGlobal_PIX_NY_Tower;
  }
  LOG(debug) << " tower size is set to : " << mTowerSizeX << " : " << mTowerSizeY << " : wafer size = " << mWaferSizeX << " : " << mWaferSizeY;
}

//_________________________________________________________________________
bool Geometry::disabledTower(int tower)
{
  return std::find(mDisableTowers.begin(), mDisableTowers.end(), tower) != mDisableTowers.end();
}

//_________________________________________________________________________
/// this gives global position of the pixel
std::tuple<double, double, double> Geometry::getGeoPixelCenter(int pixel, int tower, int layer, int stack, int row, int col) const
{
  auto [x0, y0, z0] = getGeoPadCenter(tower, layer, stack, row, col);

  int pixel_y = pixel & 0xff;
  int pixel_x = (pixel >> 8) & 0xff;

  double x1, y1;
  x1 = pixel_x * mGlobal_Pixel_Size + 0.5 * mGlobal_Pixel_Size - 0.5 * mGlobal_Pad_Size;
  y1 = pixel_y * mGlobal_Pixel_Size + 0.5 * mGlobal_Pixel_Size - 0.5 * mGlobal_Pad_Size;

  return {x1 + x0, y1 + y0, z0};
}

std::tuple<bool, int, int, int, int> Geometry::getVirtualInfo(double x, double y, double z) const
{
  //
  // Calculate col, row, layer, (virtual) segment from x,y,z
  // returns false if outside volume
  //
  int col = -1, row = -1;
  auto [status, layer, segment] = getVirtualLayerSegment(z);

  if (!status) {
    return {false, col, row, layer, segment};
  }
  if (segment == -1) {
    return {false, col, row, layer, segment};
  }
  if (std::abs(x) > (getFOCALSizeX() + 2 * getMiddleTowerOffset()) / 2) {
    return {false, col, row, layer, segment};
  }
  if (std::abs(y) > getFOCALSizeY() / 2) {
    return {false, col, row, layer, segment};
  }

  if (getVirtualIsHCal(segment)) {
    float towerSize = getHCALTowerSize();
    double beamPipeRadius = 3.0; // in cm   TODO check the number is OK (different hardcoded values are used elsewhere)
    double minRadius = beamPipeRadius + towerSize / 2.;

    double hCALsizeX = getHCALTowersInX() * towerSize;
    double hCALsizeY = getHCALTowersInY() * towerSize;

    if (x < minRadius && x > -minRadius && y < minRadius && y > -minRadius) {
      x = x < 0 ? x - 0.001 : x + 0.001;
      y = y < 0 ? y - 0.001 : y + 0.001;
    }
    if (!mUseSandwichHCAL) {
      row = (int)((y + hCALsizeY / 2) / (towerSize / 7));
      col = (int)((x + hCALsizeX / 2) / (towerSize / 7));
    } else {
      row = (int)((y + hCALsizeY / 2) / (towerSize));
      col = (int)((x + hCALsizeX / 2) / (towerSize));
    }
  } else {
    row = (int)((y + getFOCALSizeY() / 2) / mVirtualSegmentComposition[segment].mPadSize);
    // if it is the towers right and left of the beam pipe, adjust x so the offset is removed
    // if(y < 4.2 && y > - 4.2) { // TO BE set from outside or somewhere else -4,4 is the y position of the middle towers
    //     x = x < 0 ? x + GetMiddleTowerOffset() : x - GetMiddleTowerOffset();
    // }
    col = (int)((x + getFOCALSizeX() / 2) / mVirtualSegmentComposition[segment].mPadSize);
  }
  return {true, col, row, layer, segment};
}

//_______________________________________________________________________
std::tuple<bool, double, double, double> Geometry::getXYZFromColRowSeg(int col, int row, int segment) const
{

  double x = 0.0, y = 0.0, z = 0.0;
  if (segment > mVirtualNSegments) {
    return {false, x, y, z};
  }

  if (getVirtualIsHCal(segment)) {
    float towerSize = getHCALTowerSize();
    double hCALsizeX = getHCALTowersInX() * towerSize;
    double hCALsizeY = getHCALTowersInY() * towerSize;

    if (!mUseSandwichHCAL) {
      y = -1 * hCALsizeY / 2 + ((float)row + 0.5) * (towerSize / 7);
      x = -1 * hCALsizeX / 2 + ((float)col + 0.5) * (towerSize / 7);
    } else {
      y = -1 * hCALsizeY / 2 + ((float)row + 0.5) * (towerSize);
      x = -1 * hCALsizeX / 2 + ((float)col + 0.5) * (towerSize);
    }
  } else {
    y = -1 * getFOCALSizeY() / 2 + ((float)row + 0.5) * mVirtualSegmentComposition[segment].mPadSize;
    x = -1 * getFOCALSizeX() / 2 + ((float)col + 0.5) * mVirtualSegmentComposition[segment].mPadSize;
    // Middle towers offset
    // if(y < 4.2 && y > - 4.2) { // TO BE set from outside or somewhere else -4,4 is the y position of the middle towers
    //     x = x < 0 ? x - GetMiddleTowerOffset() : x + GetMiddleTowerOffset();
    // }
  }

  if (std::abs(x) > (getFOCALSizeX() + 2 * getMiddleTowerOffset()) / 2) {
    return {false, x, y, z};
  }
  if (std::abs(y) > getFOCALSizeY() / 2) {
    return {false, x, y, z};
  }
  z = getVirtualSegmentZ(segment);
  return {true, x, y, z};
}

//_______________________________________________________________________
std::tuple<bool, int, int> Geometry::getVirtualNColRow(int segment) const
{

  // ix + iy*mGlobal_PAD_NX + itowerX*mGlobal_PAD_NX*mGlobal_PAD_NY + itowerY*mGlobal_PAD_NX_Tower*mGlobal_PAD_NX*mGlobal_PAD_NY
  int nCol = -1, nRow = -1;
  if (mVirtualSegmentComposition.size() == 0) {
    return {false, nCol, nRow};
  }
  if ((segment < 0) || (segment >= mVirtualNSegments)) {
    return {false, nCol, nRow};
  }
  nCol = (int)(getFOCALSizeX() / mVirtualSegmentComposition[segment].mPadSize + 0.001);
  nRow = (int)(getFOCALSizeY() / mVirtualSegmentComposition[segment].mPadSize + 0.001);
  if (getVirtualIsHCal(segment)) {
    if (!mUseSandwichHCAL) {
      nCol = getHCALTowersInX() * 7; // To be set from outside (number of channels in each tower on x)
      nRow = getHCALTowersInY() * 7; // To be set from outside (number of channels in each tower on y)
    } else {
      nCol = getHCALTowersInX();
      nRow = getHCALTowersInY();
    }
  }
  return {true, nCol, nRow};
}

//_______________________________________________________________________
int Geometry::getVirtualNSegments() const
{

  return mVirtualNSegments;
}

//_______________________________________________________________________
std::tuple<bool, int, int> Geometry::getVirtualLayerSegment(float z) const
{

  int layer = -1;
  int segment = -1;

  z = z - getFOCALZ0() + getFOCALSizeZ() / 2;                                                // z from front face (excluding fron matter)
  float emLayersZ = mNPadLayers * mPadLayerThickness + mNPixelLayers * mPixelLayerThickness; // Pixel layers replace pad layers
  if (z < emLayersZ) {
    layer = mNPadLayers + mNPixelLayers - 1;
    while (layer >= 0 && z < mLocalLayerZ[layer]) {
      layer--;
    }
  } else {
    z = z - emLayersZ;
    layer = int(z / mHCalLayerThickness) + mNPadLayers + mNPixelLayers;
  }

  if ((layer < 0) || (layer >= (mNPadLayers + mNPixelLayers + mNHCalLayers))) {
    return {false, layer, segment};
  }

  segment = -1;
  for (int nSeg = 0; nSeg < mVirtualNSegments; nSeg++) {
    if ((layer >= mVirtualSegmentComposition[nSeg].mMinLayer) && (layer <= mVirtualSegmentComposition[nSeg].mMaxLayer)) {
      segment = nSeg;
      break;
    }
  }

  if (segment == mVirtualNSegments) {
    return {false, layer, segment};
  }
  return {true, layer, segment};
}

//_______________________________________________________________________
std::tuple<bool, int> Geometry::getVirtualSegmentFromLayer(int layer) const
{

  int segment = -1;
  for (int nSeg = 0; nSeg < mVirtualNSegments; nSeg++) {
    // cout << "Segment boundaries " << nSeg << " : " << mVirtualSegmentComposition[nSeg].fMinLayer << " " << mVirtualSegmentComposition[nSeg].fMaxLayer << endl;
    if ((layer >= mVirtualSegmentComposition[nSeg].mMinLayer) && (layer <= mVirtualSegmentComposition[nSeg].mMaxLayer)) {
      segment = nSeg;
      break;
    }
  }
  if (segment == mVirtualNSegments) {
    return {false, segment};
  }
  return {true, segment};
}

//_______________________________________________________________________
int Geometry::getVirtualSegment(float z) const
{
  auto [status, layer, segment] = getVirtualLayerSegment(z);
  return segment;
}

//_______________________________________________________________________
float Geometry::getVirtualPadSize(int segment) const
{
  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }
  return mVirtualSegmentComposition[segment].mPadSize;
}

//_______________________________________________________________________
float Geometry::getVirtualRelativeSensitiveThickness(int segment) const
{

  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }
  return mVirtualSegmentComposition[segment].mRelativeSensitiveThickness;
}

//_______________________________________________________________________
float Geometry::getVirtualPixelTreshold(int segment) const
{

  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }
  return mVirtualSegmentComposition[segment].mPixelTreshold;
}

//________________________________________________________________________
float Geometry::getVirtualSegmentSizeZ(int segment) const
{

  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }

  float size = 0;
  for (int nLay = mVirtualSegmentComposition[segment].mMinLayer; nLay <= mVirtualSegmentComposition[segment].mMaxLayer; nLay++) {
    size += mLayerThickness[nLay];
  }
  return size;
}

//________________________________________________________________________
float Geometry::getVirtualSegmentZ(int segment) const
{

  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }

  float before = 0;
  float thickness = 0;

  for (int nLay = 0; nLay < mVirtualSegmentComposition[segment].mMinLayer; nLay++) {
    before += mLayerThickness[nLay];
  }
  for (int nLay = mVirtualSegmentComposition[segment].mMinLayer; nLay <= mVirtualSegmentComposition[segment].mMaxLayer; nLay++) {
    thickness += mLayerThickness[nLay];
  }
  return getFOCALZ0() - getFOCALSizeZ() / 2 + before + thickness / 2;
}

//________________________________________________________________________
bool Geometry::getVirtualIsPixel(int segment) const
{

  if (mVirtualSegmentComposition.size() == 0) {
    return false;
  }

  if ((segment < 0) || (segment >= mVirtualNSegments)) {
    return false;
  }

  return (mVirtualSegmentComposition[segment].mIsPixel == 1);
}

//________________________________________________________________________
bool Geometry::getVirtualIsHCal(int segment) const
{

  if (mVirtualSegmentComposition.size() == 0) {
    return false;
  }

  if ((segment < 0) || (segment >= mVirtualNSegments)) {
    return false;
  }
  return (mVirtualSegmentComposition[segment].mIsPixel == 2);
}

//________________________________________________________________________
int Geometry::getVirtualNLayersInSegment(int segment) const
{
  //
  // Get the number of layers in a given segment
  //
  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }

  if ((segment < 0) || (segment >= mVirtualNSegments)) {
    return -1;
  }
  return (mVirtualSegmentComposition[segment].mMaxLayer - mVirtualSegmentComposition[segment].mMinLayer + 1);
}

//_______________________________________________________________________
int Geometry::getVirtualMinLayerInSegment(int segment) const
{
  //
  // Get the number of first layer in a given segment
  //
  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }

  if ((segment < 0) || (segment >= mVirtualNSegments)) {
    return -1;
  }
  return mVirtualSegmentComposition[segment].mMinLayer;
}

//_______________________________________________________________________
int Geometry::getVirtualMaxLayerInSegment(int segment) const
{
  //
  // Get the number of first layer in a given segment
  //
  if (mVirtualSegmentComposition.size() == 0) {
    return -1;
  }

  if ((segment < 0) || (segment >= mVirtualNSegments)) {
    return -1;
  }
  return mVirtualSegmentComposition[segment].mMaxLayer;
}
