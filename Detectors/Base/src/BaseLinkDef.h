// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ enum o2::Base::TransformType;
 
#pragma link C++ class o2::Base::Detector+;
#pragma link C++ class o2::Base::Track::TrackPar+;
#pragma link C++ class o2::Base::Track::TrackParCov+;
#pragma link C++ class o2::Base::Track::Propagator+;

#pragma link C++ class o2::Base::TrackReference+;
#pragma link C++ class o2::Base::DetID+;
#pragma link C++ class o2::Base::GeometryManager+;
#pragma link C++ class o2::Base::GeometryManager::MatBudget+;
#pragma link C++ class o2::Base::BaseCluster<float>+;
#pragma link C++ class o2::Base::MatrixCache<o2::Base::Transform3D>+;
#pragma link C++ class o2::Base::MatrixCache<o2::Base::Rotation2D>+;
#pragma link C++ class o2::Base::DetMatrixCache+;
#pragma link C++ class o2::Base::MaterialManager+;

#endif
