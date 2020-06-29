// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_RECODECAY_H_
#define O2_ANALYSIS_RECODECAY_H_

float energy(float px, float py, float pz, float mass)
{
  float en_ = sqrtf(mass * mass + px * px + py * py + pz * pz);
  return en_;
};

float invmass2prongs2(float px0, float py0, float pz0, float mass0,
                      float px1, float py1, float pz1, float mass1)
{

  float energy0_ = energy(px0, py0, pz0, mass0);
  float energy1_ = energy(px1, py1, pz1, mass1);
  float energytot = energy0_ + energy1_;

  float psum2 = (px0 + px1) * (px0 + px1) +
                (py0 + py1) * (py0 + py1) +
                (pz0 + pz1) * (pz0 + pz1);
  return energytot * energytot - psum2;
};

float invmass3prongs2(float px0, float py0, float pz0, float mass0,
                      float px1, float py1, float pz1, float mass1,
                      float px2, float py2, float pz2, float mass2)
{
  float energy0_ = energy(px0, py0, pz0, mass0);
  float energy1_ = energy(px1, py1, pz1, mass1);
  float energy2_ = energy(px2, py2, pz2, mass2);
  float energytot = energy0_ + energy1_ + energy2_;

  float psum2 = (px0 + px1 + px2) * (px0 + px1 + px2) +
                (py0 + py1 + py2) * (py0 + py1 + py2) +
                (pz0 + pz1 + pz2) * (pz0 + pz1 + pz2);
  return energytot * energytot - psum2;
};

float invmass2prongs(float px0, float py0, float pz0, float mass0,
                     float px1, float py1, float pz1, float mass1)
{
  return sqrt(invmass2prongs2(px0, py0, pz0, mass0,
                              px1, py1, pz1, mass1));
};

float invmass3prongs(float px0, float py0, float pz0, float mass0,
                     float px1, float py1, float pz1, float mass1,
                     float px2, float py2, float pz2, float mass2)
{
  return sqrt(invmass3prongs2(px0, py0, pz0, mass0,
                              px1, py1, pz1, mass1,
                              px2, py2, pz2, mass2));
};

float ptcand2prong(float px0, float py0, float px1, float py1)
{
  return sqrt((px0 + px1) * (px0 + px1) + (py0 + py1) * (py0 + py1));
};

float pttrack(float px, float py)
{
  return sqrt(px * px + py * py);
};

float declength(float xdecay, float ydecay, float zdecay, float xvtx, float yvtx, float zvtx)
{
  return sqrtf((xdecay - xvtx) * (xdecay - xvtx) + (ydecay - yvtx) * (ydecay - yvtx) + (zdecay - zvtx) * (zdecay - zvtx));
};

float declengthxy(float xdecay, float ydecay, float xvtx, float yvtx)
{
  return sqrtf((xdecay - xvtx) * (xdecay - xvtx) + (ydecay - yvtx) * (ydecay - yvtx));
};

#endif // O2_ANALYSIS_RECODECAY_H_
