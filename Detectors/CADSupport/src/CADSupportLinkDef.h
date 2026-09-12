// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Sandro Wenzel <sandro.wenzel@cern.ch>
/// \since 2026-09

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::cad::BVHSurfaceCurveRecord + ;
#pragma link C++ class o2::cad::BVHSurfaceRecord + ;
#pragma link C++ class std::vector < o2::cad::BVHSurfaceCurveRecord> + ;
#pragma link C++ class std::vector < o2::cad::BVHSurfaceRecord> + ;
#pragma link C++ class o2::cad::O2BVHSurfaceSolid - ;
#pragma link C++ class o2::cad::O2BVHAssembly + ;
#pragma link C++ class o2::cad::FlatCSGHalfspace + ;
#pragma link C++ class o2::cad::FlatCSGCell + ;
#pragma link C++ class std::vector < o2::cad::FlatCSGHalfspace> + ;
#pragma link C++ class std::vector < o2::cad::FlatCSGCell> + ;
#pragma link C++ class o2::cad::O2FlatCSG + ;
// Close every O2FlatCSG read from a file, so that any reader gets the accelerated shape.
#pragma read sourceClass = "o2::cad::O2FlatCSG" targetClass = "o2::cad::O2FlatCSG" version = "[1-]" source = "" target = "" code = "{ newObj->CloseShape(); if (!newObj->IsClosed()) { newObj->Error(\"Streamer\", \"Shape %s was read from a file and CloseShape() refused it, so it has no sub-cell boxes and every query falls back to its _Loop twin. See the Error above: a cell bounding box is missing, inverted or non-finite.\", newObj->GetName()); } }";

#endif
