/// \file TrackP
/// \brief Base track model for the Barrel, params only, w/o covariance
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_TRACK
#define ALICEO2_BASE_TRACK

#include <iostream>
#include <algorithm>
#include <string.h>
#include "DetectorsBase/Constants.h"
#include "DetectorsBase/Utils.h"

namespace AliceO2 {
  namespace Base {
    namespace Track {

      using namespace AliceO2::Base::Constants;
      using namespace AliceO2::Base::Utils;
      using namespace std;

      // aliases for track elements
      enum ParLabels : int {
        kY,kZ,kSnp,kTgl,kQ2Pt
      };
      enum CovLabels : int {
        kSigY2,
        kSigZY,kSigZ2,
        kSigSnpY,kSigSnpZ,kSigSnp2,
        kSigTglY,kSigTglZ,kSigTglSnp,kSigTgl2,
        kSigQ2PtY,kSigQ2PtZ,kSigQ2PtSnp,kSigQ2PtTgl,kSigQ2Pt2
      };

      constexpr int
        kNParams=5,
        kCovMatSize=15,
        kLabCovMatSize=21;

      constexpr float
        kCY2max = 100*100, // SigmaY<=100cm
        kCZ2max = 100*100, // SigmaZ<=100cm
        kCSnp2max = 1*1,     // SigmaSin<=1
        kCTgl2max = 1*1,     // SigmaTan<=1
        kC1Pt2max = 100 * 100, // Sigma1/Pt<=100 1/GeV
        kCalcdEdxAuto = -999.f ; // value indicating request for dedx calculation

      // helper function
      float BetheBlochSolid(float bg, float rho=2.33f,float kp1=0.20f,float kp2=3.00f,
			    float meanI=173e-9f,float meanZA=0.49848f);
      void g3helx3(float qfield, float step,array<float,7> &vect);


      class TrackParBase { // track parameterization, kinematics only. This base class cannot be instantiated
        public:

          ///const float* GetParam()              const { return mP; }
          float GetX()                         const { return mX; }
          float GetAlpha()                     const { return mAlpha; }
          float GetY()                         const { return mP[kY]; }
          float GetZ()                         const { return mP[kZ]; }
          float GetSnp()                       const { return mP[kSnp]; }
          float GetTgl()                       const { return mP[kTgl]; }
          float GetQ2Pt()                      const { return mP[kQ2Pt]; }

          // derived getters
          float GetCurvature(float b)          const { return mP[kQ2Pt]*b*kB2C;}
          float GetSign()                      const { return mP[kQ2Pt]>0 ? 1.f:-1.f;}
          float GetPhi()                       const { return asinf(GetSnp()) + GetAlpha();}
          float GetPhiPos()                    const;

          float GetP()                         const;
          float GetPt()                        const;
          void  GetXYZ(array<float,3> &xyz)           const;
          bool  GetPxPyPz(array<float,3> &pxyz)       const;
          bool  GetPosDir(array<float,9> &posdirp)    const;

          // parameters manipulation
          bool  RotateParam(float alpha);
          bool  PropagateParamTo(float xk, float b);
          bool  PropagateParamTo(float xk, const array<float,3> &b);
          void  InvertParam();

          void  PrintParam()                   const;

        protected:
          // to keep this class non-virtual but derivable the c-tors and d-tor are protected
          TrackParBase() : mX{0.},mAlpha{0.} {}
          TrackParBase(float x,float alpha, const array<float,kNParams> &par);
          TrackParBase(const array<float,3> &xyz,const array<float,3> &pxpypz, int sign, bool sectorAlpha=true);
          TrackParBase(const TrackParBase&) = default;
          TrackParBase(TrackParBase&&) = default;
          TrackParBase& operator=(const TrackParBase& src) = default;
          ~TrackParBase() = default;
          //
          float mX;                   /// X of track evaluation
          float mAlpha;               /// track frame angle
          float mP[kNParams] = {0.f}; /// 5 parameters: Y,Z,sin(phi),tg(lambda),q/pT
      };

      // rootcint does not swallow final keyword here
      class TrackParCov final : public TrackParBase { // track+error parameterization
        public:
          TrackParCov() : TrackParBase{} { }
          TrackParCov(float x, float alpha, const array<float,kNParams> &par, const array<float,kCovMatSize> &cov);
          TrackParCov(const array<float,3> &xyz,const array<float,3> &pxpypz,const array<float,kLabCovMatSize> &cv, int sign, bool sectorAlpha=true);

          ///const float* GetCov()                const { return mC; }
          float GetSigmaY2()                   const { return mC[kSigY2]; }
          float GetSigmaZY()                   const { return mC[kSigZY]; }
          float GetSigmaZ2()                   const { return mC[kSigZ2]; }
          float GetSigmaSnpY()                 const { return mC[kSigSnpY]; }
          float GetSigmaSnpZ()                 const { return mC[kSigSnpZ]; }
          float GetSigmaSnp2()                 const { return mC[kSigSnp2]; }
          float GetSigmaTglY()                 const { return mC[kSigTglY]; }
          float GetSigmaTglZ()                 const { return mC[kSigTglZ]; }
          float GetSigmaTglSnp()               const { return mC[kSigTglSnp]; }
          float GetSigmaTgl2()                 const { return mC[kSigTgl2]; }
          float GetSigma1PtY()                 const { return mC[kSigQ2PtY]; }
          float GetSigma1PtZ()                 const { return mC[kSigQ2PtZ]; }
          float GetSigma1PtSnp()               const { return mC[kSigQ2PtSnp]; }
          float GetSigma1PtTgl()               const { return mC[kSigQ2PtTgl]; }
          float GetSigma1Pt2()                 const { return mC[kSigQ2Pt2]; }

          void  Print()                        const;

          // parameters + covmat manipulation
          bool  Rotate(float alpha);
          bool  PropagateTo(float xk, float b);
          bool  PropagateTo(float xk, const array<float,3> &b);
          void  Invert();

          float GetPredictedChi2(const array<float,2> &p, const array<float,3> &cov) const;
          bool  Update(const array<float,2> &p, const array<float,3> &cov);

          bool  CorrectForMaterial(float x2x0,float xrho,float mass,bool anglecorr=false,float dedx=kCalcdEdxAuto);

          void  ResetCovariance(float s2=0);
          void  CheckCovariance();

        protected:
          float mC[kCovMatSize] = {0.f};  // 15 covariance matrix elements

      };

      class TrackPar final : public TrackParBase { // track parameterization only
        public:
          TrackPar() {}
          TrackPar(float x,float alpha, const array<float,kNParams> &par) : TrackParBase{x,alpha,par} {}
          TrackPar(const array<float,3> &xyz, const array<float,3> &pxpypz,int sign, bool sectorAlpha=true);
          //
          void  Print() const {PrintParam();}
      };

      //____________________________________________________________
      inline TrackParBase::TrackParBase(float x, float alpha, const array<float, kNParams> &par) : mX{x}, mAlpha{alpha} {
        // explicit constructor
        std::copy(par.begin(), par.end(), mP);
      }

      //_______________________________________________________
      inline void TrackParBase::GetXYZ(array<float,3> &xyz) const {
        // track coordinates in lab frame
        xyz[0] = GetX();
        xyz[1] = GetY();
        xyz[2] = GetZ();
        RotateZ(xyz,GetAlpha());
      }

      //_______________________________________________________
      inline float TrackParBase::GetPhiPos() const {
        // angle of track position
        float xy[2]={GetX(),GetY()};
        return atan2(xy[1],xy[0]);
      }

      //____________________________________________________________
      inline float TrackParBase::GetP() const {
        // return the track momentum
        float ptI = fabs(GetQ2Pt());
        return (ptI>kAlmost0) ? sqrtf(1.f+ GetTgl()*GetTgl())/ptI : kVeryBig;
      }

      //____________________________________________________________
      inline float TrackParBase::GetPt() const {
        // return the track transverse momentum
        float ptI = fabs(GetQ2Pt());
        return (ptI>kAlmost0) ? 1.f/ptI : kVeryBig;
      }

      //============================================================

      //____________________________________________________________
      inline TrackParCov::TrackParCov(float x, float alpha, const array<float,kNParams> &par, const array<float,kCovMatSize> &cov)
	: TrackParBase{x,alpha,par} {
        // explicit constructor
        std::copy(cov.begin(), cov.end(), mC);
      }

      //============================================================

      //____________________________________________________________
      inline TrackPar::TrackPar(const array<float,3> &xyz, const array<float,3> &pxpypz,int sign, bool sectorAlpha)
	: TrackParBase{xyz,pxpypz,sign,sectorAlpha} {
        // explicit constructor
      }

    }
  }
}


#endif
