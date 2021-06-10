// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ANALYSIS_DIFFCUT_HOLDER_H_
#define O2_ANALYSIS_DIFFCUT_HOLDER_H_

#include <iosfwd>
#include <Rtypes.h>
#include <TMath.h>

// object to hold customizable cut values
class cutHolder
{
 public:
  // constructor
  cutHolder(bool isRun2 = false,
            bool isMC = false,
            int MinNTracks = 0, int MaxNTracks = 10000,
            int MinNTracksWithTOFHit = 0,
            int deltaBC = 7,
            float MinPosz = -1000., float MaxPosz = 1000.,
            float minPt = 0., float maxPt = 1000.,
            float minEta = -1.0, float maxEta = 1.0,
            float maxTOFChi2 = 10.,
            float maxnSigmaTPC = 3., float maxnSigmaTOF = 3.) : misMC{isMC}, mMinNTracks{MinNTracks}, mMaxNTracks{MaxNTracks}, mMinNTracksWithTOFHit{MinNTracksWithTOFHit}, mMinVertexPosz{MinPosz}, mMaxVertexPosz{MaxPosz}, mMinPt{minPt}, mMaxPt{maxPt}, mMinEta{minEta}, mMaxEta{maxEta}, mdeltaBC{deltaBC}, mMaxTOFChi2{maxTOFChi2}, mMaxnSigmaTPC{maxnSigmaTPC}, mMaxnSigmaTOF{maxnSigmaTOF}
  {
  }

  // setter
  void SetisRun2(bool isRun2);
  void SetisMC(bool isMC);
  void SetNTracks(int MinNTracks, int MaxNTracks);
  void SetMinNTracksWithTOFHit(int MinNTracksWithTOFHit);
  void SetDeltaBC(int deltaBC);
  void SetPoszRange(float MinPosz, float MaxPosz);
  void SetPtRange(float minPt, float maxPt);
  void SetEtaRange(float minEta, float maxEta);
  void SetMaxTOFChi2(float maxTOFChi2);
  void SetMaxnSigmaTPC(float maxnSigma);
  void SetMaxnSigmaTOF(float maxnSigma);

  // getter
  bool isRun2() const;
  bool isMC() const;
  int minNTracks() const;
  int maxNTracks() const;
  int minNTracksWithTOFHit() const;
  int deltaBC() const;
  float minPosz() const;
  float maxPosz() const;
  float minPt() const;
  float maxPt() const;
  float minEta() const;
  float maxEta() const;
  float maxTOFChi2() const;
  float maxnSigmaTPC() const;
  float maxnSigmaTOF() const;

 private:
  bool misRun2;

  // data or MC
  bool misMC;

  // number of tracks
  int mMinNTracks, mMaxNTracks; // Number of allowed tracks

  // number of tracks with TOF hit
  int mMinNTracksWithTOFHit;

  // BC range for past-future protection
  int mdeltaBC;

  // vertex z-position
  float mMinVertexPosz, mMaxVertexPosz; // Vertex z-position

  // kinematic cuts
  float mMinPt, mMaxPt;   // range in track pT
  float mMinEta, mMaxEta; // range in track eta

  // maximum TOF chi2
  float mMaxTOFChi2; // maximum TOF Chi2

  // maximum nSigma for PID
  float mMaxnSigmaTPC; // maximum nSigma TPC
  float mMaxnSigmaTOF; // maximum nSigma TOF

  ClassDef(cutHolder, 1);
};

#endif // O2_ANALYSIS_DIFFCUT_HOLDER_H_
