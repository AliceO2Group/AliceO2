// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file UtilsDebugLcK0Sp
/// \brief Some utilities to do debugging for the LcK0Sp task
///
/// For example, for: /alice/sim/2020/LHC20l3a/286350/PWGZZ/Run3_Conversion/156_20210308-1000/0001/AO2D.root
///
/// listLabelsProton = {717, 2810, 4393, 5442, 6769, 7793, 9002, 9789, 10385, 12601, 13819, 17443, 19936, 21715, 23285, 23999, 25676, 28660, 30684, 32049, 34846, 37042, 39278, 40278, 41567, 43696, 44819, 45232, 46578, 47831, 48981, 53527, 57152, 58155, 59296, 61010, 61932, 63577, 65683, 68047, 69657, 70721, 73141, 75320, 76581, 78165, 79504, 80662, 81552, 84208, 86342, 86829, 87347, 89922, 95032, 96087, 97410, 100154, 101743, 103368, 104635, 106089, 108676, 110308, 111335, 111342, 112606, 113469, 114529, 114533, 116388, 117754, 118765, 118784, 119733, 120691, 121773, 123970, 125316, 127253, 129008, 129883, 131408, 132349, 132395, 133263, 134063, 135417, 136194, 137441, 138569, 139167, 141741, 143422, 144322, 145013, 145695, 147134, 148503, 149283, 149728, 153214, 154481, 155193, 158187, 159039, 160009, 161126, 162191, 163774, 165792, 166934, 168768, 174013, 174017, 175467, 177005, 177258, 178380, 179566, 179577, 181145, 187316, 188512, 189094, 191582, 193018, 194159, 195821, 197459, 201840, 202646, 203119, 203763, 205553, 207745, 208851, 211636, 216231, 217125, 217516, 218522, 219477, 219960, 223246, 224677, 224702, 225438, 227194, 230507, 231304, 232070, 232772, 234765, 235877, 236893, 237989, 239575, 241469, 243404, 244872, 245511, 246688, 249625, 250580, 251879, 253031, 254465, 254511, 255917, 256782, 258734, 261436, 262878, 264465, 264467, 268907, 269974, 271856, 274044, 276071, 276915, 278461, 279559, 280441, 281783, 281976, 283405, 284722, 286324, 287929, 289681, 291005, 292324, 293478, 296484, 300536}
/// listLabelsK0SPos = {729, 2866, 4754, 5457, 6891, 7824, 9243, 9810, 10388, 12665, 13830, 17460, 19955, 21786, 23319, 24010, 26234, 28713, 30699, 32056, 34866, 37075, 39314, 40287, 41617, 43790, 44868, 45252, 46587, 47866, 48992, 53534, 57211, 58160, 59355, 61019, 62003, 63580, 65691, 68164, 69721, 70852, 73167, 75331, 76641, 78188, 79595, 80678, 81613, 84311, 86376, 86840, 87374, 89998, 95078, 96094, 97425, 100157, 101788, 103468, 104646, 106155, 108696, 110388, 111355, 111348, 112712, 113578, 114556, 114550, 116432, 117766, 118796, 118790, 119766, 120728, 121872, 124027, 125407, 127284, 129068, 129924, 131453, 132456, 132436, 133266, 134084, 135470, 136207, 137446, 138616, 139217, 141832, 143513, 144416, 145018, 145702, 147139, 148508, 149290, 149737, 153271, 154515, 155206, 158200, 159059, 160082, 161140, 162208, 163777, 165874, 167001, 168852, 174033, 174027, 175501, 177396, 177265, 178383, 179595, 179580, 181168, 187326, 188608, 189128, 191591, 193025, 194170, 195824, 197472, 201905, 202664, 203163, 203835, 205556, 207838, 208866, 211654, 216312, 217132, 217554, 218616, 219503, 219996, 223354, 224713, 224707, 225482, 227207, 230562, 231324, 232083, 232809, 234774, 235892, 236900, 238038, 239629, 241517, 243460, 244881, 245553, 246721, 249672, 250589, 251882, 253034, 254543, 254539, 255941, 256808, 258807, 261612, 262887, 264473, 264471, 268912, 270023, 271998, 274081, 276125, 276952, 278530, 279613, 280446, 282152, 282034, 283408, 284753, 286615, 288037, 289736, 291009, 292336, 293565, 296529, 300564}
/// listLabelsK0SNeg = {730, 2867, 4755, 5458, 6892, 7825, 9244, 9811, 10389, 12666, 13831, 17461, 19956, 21787, 23320, 24011, 26235, 28714, 30700, 32057, 34867, 37076, 39315, 40288, 41618, 43791, 44869, 45253, 46588, 47867, 48993, 53535, 57212, 58161, 59356, 61020, 62004, 63581, 65692, 68165, 69722, 70853, 73168, 75332, 76642, 78189, 79596, 80679, 81614, 84312, 86377, 86841, 87375, 89999, 95079, 96095, 97426, 100158, 101789, 103469, 104647, 106156, 108697, 110389, 111356, 111349, 112713, 113579, 114557, 114551, 116433, 117767, 118797, 118791, 119767, 120729, 121873, 124028, 125408, 127285, 129069, 129925, 131454, 132457, 132437, 133267, 134085, 135471, 136208, 137447, 138617, 139218, 141833, 143514, 144417, 145019, 145703, 147140, 148509, 149291, 149738, 153272, 154516, 155207, 158201, 159060, 160083, 161141, 162209, 163778, 165875, 167002, 168853, 174034, 174028, 175502, 177397, 177266, 178384, 179596, 179581, 181169, 187327, 188609, 189129, 191592, 193026, 194171, 195825, 197473, 201906, 202665, 203164, 203836, 205557, 207839, 208867, 211655, 216313, 217133, 217555, 218617, 219504, 219997, 223355, 224714, 224708, 225483, 227208, 230563, 231325, 232084, 232810, 234775, 235893, 236901, 238039, 239630, 241518, 243461, 244882, 245554, 246722, 249673, 250590, 251883, 253035, 254544, 254540, 255942, 256809, 258808, 261613, 262888, 264474, 264472, 268913, 270024, 271999, 274082, 276126, 276953, 278531, 279614, 280447, 282153, 282035, 283409, 284754, 286616, 288038, 289737, 291010, 292337, 293566, 296530, 300565}
///

#include <vector>
#include <algorithm>

inline bool isK0SfromLcFunc(int labelK0SPos, int labelK0SNeg, std::vector<int> listLabelsK0SPos, std::vector<int> listLabelsK0SNeg)
{

  auto nPositiveDau = listLabelsK0SPos.size();
  auto nNegativeDau = listLabelsK0SNeg.size();

  // checking sizes of vectors: they should be identical
  if (nPositiveDau != nNegativeDau) {
    LOG(ERROR) << "Number of elements in vector of positive daughters of K0S different from the number of elements in the vector for the negative ones: " << nPositiveDau << " : " << nNegativeDau;
    throw std::runtime_error("sizes of configurables for debug do not match");
  }

  // checking if we find the candidate
  bool matchesK0Spositive = std::any_of(listLabelsK0SPos.begin(), listLabelsK0SPos.end(), [&labelK0SPos](const int& label) { return label == labelK0SPos; });
  bool matchesK0Snegative = std::any_of(listLabelsK0SNeg.begin(), listLabelsK0SNeg.end(), [&labelK0SNeg](const int& label) { return label == labelK0SNeg; });
  if (matchesK0Spositive && matchesK0Snegative) {
    return true;
  }

  return false;
}

//-------------------------------

inline bool isProtonFromLcFunc(int labelProton, std::vector<int> listLabelsProton)
{
  // checking if we find the candidate`
  bool matchesProton = std::any_of(listLabelsProton.begin(), listLabelsProton.end(), [&labelProton](const int& label) { return label == labelProton; });
  if (matchesProton) {
    return true;
  }
  return false;
}

//---------------------------------

inline bool isLcK0SpFunc(int labelProton, int labelK0SPos, int labelK0SNeg, std::vector<int> listLabelsProton, std::vector<int> listLabelsK0SPos, std::vector<int> listLabelsK0SNeg)
{

  auto nPositiveDau = listLabelsK0SPos.size();
  auto nNegativeDau = listLabelsK0SNeg.size();
  auto nProtons = listLabelsProton.size();

  // checking sizes of vectors: they should be identical
  if (nPositiveDau != nNegativeDau || nPositiveDau != nProtons) {
    LOG(ERROR) << "Number of elements in vector of positive daughters of K0S, in vector of negative daughters of K0S, and in vector of protons differ: " << nPositiveDau << " : " << nNegativeDau << " : " << nProtons;
    throw std::runtime_error("sizes of configurables for debug do not match");
  }

  // checking if we find the candidate
  bool matchesK0Spositive = std::any_of(listLabelsK0SPos.begin(), listLabelsK0SPos.end(), [&labelK0SPos](const int& label) { return label == labelK0SPos; });
  bool matchesK0Snegative = std::any_of(listLabelsK0SNeg.begin(), listLabelsK0SNeg.end(), [&labelK0SNeg](const int& label) { return label == labelK0SNeg; });
  bool matchesProton = std::any_of(listLabelsProton.begin(), listLabelsProton.end(), [&labelProton](const int& label) { return label == labelProton; });
  if (matchesK0Spositive && matchesK0Snegative && matchesProton) {
    return true;
  }

  return false;
}
