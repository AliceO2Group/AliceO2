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

/// \file GPUDisplayGUI.h
/// \author David Rohr

#ifndef GPUDISPLAYGUI_H
#define GPUDISPLAYGUI_H

#define QT_BEGIN_NAMESPACE \
  namespace o2::gpu::qtgui \
  {
#define QT_END_NAMESPACE }

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui
{
class GPUDisplayGUI;
}
QT_END_NAMESPACE

namespace o2::gpu
{
class GPUDisplayGUIWrapper;
class GPUDisplayGUI : public QMainWindow
{
  Q_OBJECT

 public:
  GPUDisplayGUI(QWidget* parent = nullptr);
  ~GPUDisplayGUI();
  void setWrapper(GPUDisplayGUIWrapper* w) { mWrapper = w; }

 private slots:
  void UpdateTimer();

 private:
  Ui::GPUDisplayGUI* ui;
  GPUDisplayGUIWrapper* mWrapper;
};
} // namespace o2::gpu

#endif // GPUDISPLAYGUI_H
