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

double energy(double px, double py, double pz, double mass)
{
  double en_ = sqrt(mass * mass + px * px + py * py + pz * pz);
  return en_;
};

double invmass2prongs2(double px0, double py0, double pz0, double mass0,
                       double px1, double py1, double pz1, double mass1)
{

  double energy0_ = energy(px0, py0, pz0, mass0);
  double energy1_ = energy(px1, py1, pz1, mass1);
  double energytot = energy0_ + energy1_;

  double psum2 = (px0 + px1) * (px0 + px1) +
                 (py0 + py1) * (py0 + py1) +
                 (pz0 + pz1) * (pz0 + pz1);
  return energytot * energytot - psum2;
};

double invmass3prongs2(double px0, double py0, double pz0, double mass0,
                       double px1, double py1, double pz1, double mass1,
                       double px2, double py2, double pz2, double mass2)
{
  double energy0_ = energy(px0, py0, pz0, mass0);
  double energy1_ = energy(px1, py1, pz1, mass1);
  double energy2_ = energy(px2, py2, pz2, mass2);
  double energytot = energy0_ + energy1_ + energy2_;

  double psum2 = (px0 + px1 + px2) * (px0 + px1 + px2) +
                 (py0 + py1 + py2) * (py0 + py1 + py2) +
                 (pz0 + pz1 + pz2) * (pz0 + pz1 + pz2);
  return energytot * energytot - psum2;
};

double invmass2prongs(double px0, double py0, double pz0, double mass0,
                      double px1, double py1, double pz1, double mass1)
{
  return sqrt(invmass2prongs2(px0, py0, pz0, mass0,
                              px1, py1, pz1, mass1));
};

double invmass3prongs(double px0, double py0, double pz0, double mass0,
                      double px1, double py1, double pz1, double mass1,
                      double px2, double py2, double pz2, double mass2)
{
  return sqrt(invmass3prongs2(px0, py0, pz0, mass0,
                              px1, py1, pz1, mass1,
                              px2, py2, pz2, mass2));
};

double ptcand2prong(double px0, double py0, double px1, double py1)
{
  return sqrt((px0 + px1) * (px0 + px1) + (py0 + py1) * (py0 + py1));
};

double pttrack(double px, double py)
{
  return sqrt(px * px + py * py);
};

double declength(double xdecay, double ydecay, double zdecay, double xvtx, double yvtx, double zvtx)
{
  return sqrt((xdecay - xvtx) * (xdecay - xvtx) + (ydecay - yvtx) * (ydecay - yvtx) + (zdecay - zvtx) * (zdecay - zvtx));
};

double declengthxy(double xdecay, double ydecay, double xvtx, double yvtx)
{
  return sqrt((xdecay - xvtx) * (xdecay - xvtx) + (ydecay - yvtx) * (ydecay - yvtx));
};

/// Add a track object to a list.
/// \param list      vector of track objects
/// \param momentum  3-momentum components (x, y, z)
/// \param temporal  mass or energy (depends on the vector type)
template <typename T, typename U, typename V>
void addTrack(std::vector<T>& list, U momentum_x, U momentum_y, U momentum_z, V temporal)
{
  T track(momentum_x, momentum_y, momentum_z, temporal);
  list.push_back(track);
}

/// Add a track object to a list.
/// \param list      vector of track objects
/// \param momentum  array of 3-momentum components (x, y, z)
/// \param temporal  mass or energy (depends on the vector type)
template <typename T, typename U, typename V>
void addTrack(std::vector<T>& list, const std::array<U, 3>& momentum, V temporal)
{
  addTrack(list, momentum[0], momentum[1], momentum[2], temporal);
}

/// Sum up track objects in a list.
/// \param list  vector of track objects
/// \return      Return the sum of the vector elements.
template <typename T>
T sumOfTracks(const std::vector<T>& list)
{
  T sum;
  for (const auto& track : list) {
    sum += track;
  }
  return sum;
}

#endif // O2_ANALYSIS_RECODECAY_H_
