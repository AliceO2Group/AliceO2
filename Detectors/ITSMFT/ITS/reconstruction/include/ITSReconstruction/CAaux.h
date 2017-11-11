// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALIITSUCACELL_H
#define ALIITSUCACELL_H

#include <vector>
#include <array>
#include <cmath>
#include "DetectorsBase/Track.h"
#include "DetectorsBase/Constants.h"

namespace o2 {
  namespace ITS {
    namespace CA {

      class Cell {
        public:
          Cell(int xx = 0u,int yy = 0u, int zz = 0u, int dd0 = 0,
              int dd1 = 0, float curv = 0.f, std::array<float,3> n = {0.f});

          int x() const { return mVector[0]; }
          int y() const { return mVector[1]; }
          int z() const { return mVector[2]; }
          int d0() const { return md0; }
          int d1() const { return md1; }
          int GetLevel() const { return mVector[3]; }
          float GetCurvature() const { return m1OverR; }
          std::array<float,3>& GetN() { return mN; }

          void SetLevel(int lev) { mVector[3] = lev; }

          int operator()(const int i) { return mVector[4 + i]; }

          std::vector<int>::size_type NumberOfNeighbours() { return (mVector.size() - 4u); }

          bool Combine(Cell &neigh, int idd);

        private:
          float m1OverR;
          int md0,md1;
          std::array<float,3> mN;
          std::vector<int> mVector;
      };

      class Road {
        public:
          Road() : Elements(), N(0) {
            ResetElements();
          }

          Road(int layer, int idd) : Elements(), N() {
            ResetElements();
            N = 1;
            Elements[layer] = idd;
          }

          Road(const Road& copy) : Elements(), N(copy.N) {
            for ( int i=0; i<5; ++i ) {
              Elements[i] = copy.Elements[i];
            }
          }

          int &operator[] (const int &i) {
            return Elements[i];
          }

          void ResetElements() {
            for ( int i=0; i<5; ++i )
              Elements[i] = -1;
            N = 0;
          }

          void AddElement(int i, int el) {
            ++N;
            Elements[i] = el;
          }

          int Elements[5];
          int N;
      };

      struct Cluster {  // cluster info, optionally XY origin at vertex
        float x,y,z,phi,r;    // lab params
        std::array<float,3> cov;
        int   zphibin; // bins is z,phi
        int   detid;          // detector index //RS ??? Do we need it?
        int   label;
        bool operator<(const Cluster &rhs) const {return zphibin < rhs.zphibin;}
        //
      };
      typedef struct Cluster ClsInfo_t;

      class Track : public o2::Base::Track::TrackParCov {
        public:
          using o2::Base::Track::TrackParCov::TrackParCov;
          Track(float x, float a, std::array<float,Base::Track::kNParams> p, std::array<float,Base::Track::kCovMatSize> c, int *cl);

          int* Clusters() { return mCl; }

          int GetLabel() const { return mLabel; }
          float GetChi2() const { return mChi2; }

          void SetLabel(int label) { mLabel = label; }
          void SetChi2(float chi) { mChi2 = chi; }

          bool Update(const Cluster& cl);
          bool GetPhiZat(float r, float bfield,float &phi, float &z) const;
        private:
          int mCl[7] = {0};
          int mLabel = -1;
          float mChi2 = 0.f;
      };

      class Doublets {
        public:
          Doublets(int xx = 0, int yy = 0, float tL = 0.f, float ph = 0.f)
            : x((int)xx)
              , y((int)yy)
              , tanL(tL)
              , phi(ph) {}
          int x,y;
          float tanL, phi;
      };

      struct Sensor  { // info on sensor
        int index; // sensor vid
        float xTF,xTFmisal,phiTF,sinTF,cosTF; //tracking frame parameters of the detector
      };
      typedef struct Sensor ITSDetInfo_t;

      inline float invsqrt(float _x) {
        //
        // The function calculates fast inverse sqrt. Google for 0x5f3759df.
        // Credits to ALICE HLT Project
        //

        union { float f; int i; } x = { _x };
        const float xhalf = 0.5f * x.f;
        x.i = 0x5f3759df - ( x.i >> 1 );
        x.f = x.f * ( 1.5f - xhalf * x.f * x.f );
        return x.f;
      }

      inline float Curvature(float x1, float y1, float x2, float y2, float x3, float y3) {
        //
        // Initial approximation of the track curvature
        //
        return   2.f * ((x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1))
          * invsqrt(((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) *
              ((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3)) *
              ((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1)));
      }

      inline float TanLambda(float x1, float y1, float x2, float y2, float z1, float z2) {
        //
        // Initial approximation of the tangent of the track dip angle
        //
        return (z1 - z2) * invsqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
      }

      inline bool CompareAngles(float alpha, float beta, float tolerance) {
        const float delta = fabs(alpha - beta);
        return (delta < tolerance || fabs(delta - o2::Base::Constants::k2PI) < tolerance);
      }
    } // namespace CA
  } // namespace ITS
} // namespace AliceO2
#endif // ALIITSUCACELL_H
