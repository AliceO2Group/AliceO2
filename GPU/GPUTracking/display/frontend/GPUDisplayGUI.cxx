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

/// \file GPUDisplayGUI.cxx
/// \author David Rohr

#include "GPUDisplayGUI.h"
#include "GPUDisplayFrontend.h"
#include "ui_GPUDisplayGUI.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleValidator>
#include <QFormLayout>
#include <QGroupBox>
#include <QHash>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QScreen>
#include <QSignalBlocker>
#include <QSlider>
#include <QStyle>
#include <QWheelEvent>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <variant>
#include <vector>

using namespace o2::gpu;

static void mergeValue(std::vector<std::pair<uint32_t, GPUDisplayGUIValue>>& values, uint32_t id, GPUDisplayGUIValue value) // Replace an earlier value of the same element
{
  auto it = std::find_if(values.begin(), values.end(), [id](const auto& v) { return v.first == id; });
  if (it != values.end()) {
    it->second = std::move(value);
  } else {
    values.emplace_back(id, std::move(value));
  }
}

namespace
{
class FlowLayout : public QLayout // Places the items in a row with their preferred width, and wraps into further rows at the right border (Qt has no such layout)
{
 public:
  FlowLayout(QWidget* parent, int hSpacingFactor, bool equalRowHeight) : QLayout(parent), mHSpacingFactor(hSpacingFactor), mEqualRowHeight(equalRowHeight) {}
  ~FlowLayout() override
  {
    while (QLayoutItem* item = takeAt(0)) {
      delete item;
    }
  }
  void addItem(QLayoutItem* item) override { mItems.append(item); }
  int count() const override { return mItems.size(); }
  QLayoutItem* itemAt(int index) const override { return mItems.value(index); }
  QLayoutItem* takeAt(int index) override { return index >= 0 && index < mItems.size() ? mItems.takeAt(index) : nullptr; }
  Qt::Orientations expandingDirections() const override { return {}; }
  bool hasHeightForWidth() const override { return true; }
  int heightForWidth(int width) const override { return doLayout(QRect(0, 0, width, 0), true); }
  void setGeometry(const QRect& rect) override
  {
    QLayout::setGeometry(rect);
    doLayout(rect, false);
  }
  QSize sizeHint() const override // All items in one row
  {
    QSize size(0, 0);
    for (const QLayoutItem* item : mItems) {
      size = QSize(size.width() + (size.width() ? hSpacing() : 0) + itemWidth(item), std::max(size.height(), item->sizeHint().height()));
    }
    const QMargins m = contentsMargins();
    return size + QSize(m.left() + m.right(), m.top() + m.bottom());
  }
  QSize minimumSize() const override // Widest item, height for that width is given by heightForWidth
  {
    QSize size;
    for (const QLayoutItem* item : mItems) {
      size = size.expandedTo(item->minimumSize());
    }
    const QMargins m = contentsMargins();
    return size + QSize(m.left() + m.right(), m.top() + m.bottom());
  }
  void setPreferredWidth(QWidget* widget, int width) { mPreferredWidth[widget] = width; } // Instead of the size hint of the widget

 private:
  int hSpacing() const { return mHSpacingFactor * spacing(); }
  int itemWidth(const QLayoutItem* item) const { return mPreferredWidth.value(item->widget(), item->sizeHint().width()); }
  int doLayout(const QRect& rect, bool testOnly) const // Returns the height needed
  {
    const QRect r = rect.marginsRemoved(contentsMargins());
    int x = r.x(), y = r.y(), lineHeight = 0;
    std::vector<std::pair<QLayoutItem*, QRect>> line; // Items of the current row, placed when the row is complete (height of the row known)
    auto placeLine = [&]() {
      if (!testOnly) {
        for (auto& [item, geometry] : line) {
          item->setGeometry(mEqualRowHeight ? QRect(geometry.x(), geometry.y(), geometry.width(), lineHeight) : geometry);
        }
      }
      line.clear();
    };
    for (QLayoutItem* item : mItems) {
      const int width = std::max(std::min(itemWidth(item), r.width()), item->minimumSize().width()); // Items wider than the layout get its width (at least their minimum), and wrap internally if they can
      const int height = item->hasHeightForWidth() ? item->heightForWidth(width) : item->sizeHint().height();
      if (x > r.x() && x + width > r.right() + 1) {
        placeLine();
        x = r.x();
        y += lineHeight + spacing();
        lineHeight = 0;
      }
      line.emplace_back(item, QRect(x, y, width, height));
      x += width + hSpacing();
      lineHeight = std::max(lineHeight, height);
    }
    placeLine();
    return y + lineHeight - rect.y() + contentsMargins().bottom();
  }
  QList<QLayoutItem*> mItems;
  QHash<const QWidget*, int> mPreferredWidth;
  int mHSpacingFactor;  // Horizontal spacing in units of spacing()
  bool mEqualRowHeight; // All items of a row get the height of the highest one
};
} // namespace

static double toDouble(const GPUDisplayGUIValue& v)
{
  return std::visit([](const auto& x) -> double {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, std::string>) {
      return QString::fromStdString(x).toDouble();
    } else {
      return x;
    }
  },
                    v);
}

static bool sameValue(const GPUDisplayGUIValue& a, const GPUDisplayGUIValue& b) // Numbers compared by value, independent of int64_t / double
{
  const bool aString = std::holds_alternative<std::string>(a), bString = std::holds_alternative<std::string>(b);
  return aString || bString ? a == b : toDouble(a) == toDouble(b);
}

GPUDisplayGUI::GPUDisplayGUI(GPUDisplayGUIChannel& channel, QWidget* parent) : QMainWindow(parent), mUi(std::make_unique<Ui::GPUDisplayGUI>()), mChannel(channel)
{
  mUi->setupUi(this);
  std::vector<GPUDisplayGUIChannel::Frame> layout;
  {
    std::lock_guard lock(mChannel.mutex);
    setWindowTitle(QString::fromStdString(mChannel.windowName + " GUI"));
    layout = mChannel.layout;
  }
  mBaseFontSize = font().pointSizeF() > 0 ? font().pointSizeF() : 10.f; // Fonts with pixel size have no point size
  buildElements(layout);
  setScale(mScale);               // Scale from the display with the first state
  qApp->installEventFilter(this); // Sliders / dropdowns would consume Ctrl + mouse wheel themselves
  connect(&mTimer, &QTimer::timeout, this, &GPUDisplayGUI::timerTick);
  mTimer.start(std::chrono::milliseconds(10));
}

GPUDisplayGUI::~GPUDisplayGUI() = default;

void GPUDisplayGUI::timerTick()
{
  std::unique_lock lock(mChannel.mutex);
  mChannel.guiSleeping = false;
  if (mUpdate.empty()) { // Display processes the update we went to sleep with, wait for the reply (with new changes we would not apply it anyway)
    mChannel.stateUpdated.wait_for(lock, mChannel.replyTimeout, [this] { return !mChannel.displayProcessingCurrentUpdate || mChannel.stop; });
  }
  if (!mUpdate.empty() && mLastSentId - mLastAppliedId >= mChannel.maxUpdatesAhead) { // Too far ahead of the display, wait for the reply to the last update
    mChannel.stateUpdated.wait_for(lock, mChannel.replyTimeout, [this] { return mChannel.state.appliedUpdate == mLastSentId || mChannel.stop; });
  }
  if (mChannel.stop) {
    lock.unlock();
    close();
    return;
  }

  const bool newState = mChannel.state.counter != mState.counter;
  if (newState) {
    mState = mChannel.state;
    if (mState.appliedUpdate == mLastSentId) {
      mLastAppliedId = mLastSentId;
    }
  }
  const bool apply = newState && mUpdate.empty() && mState.appliedUpdate == mLastSentId; // State contains all our updates, and we have no unsent changes
  const bool post = !mUpdate.empty() && mLastSentId - mLastAppliedId < mChannel.maxUpdatesAhead;
  if (post) {
    if (mChannel.update) { // Previous update not yet picked up by the display, merge
      auto& update = *mChannel.update;
      update.keys.insert(update.keys.end(), mUpdate.keys.begin(), mUpdate.keys.end());
      for (const auto& [id, value] : mUpdate.values) {
        mergeValue(update.values, id, value);
      }
      if (mUpdate.guiScale) {
        update.guiScale = mUpdate.guiScale;
      }
    } else {
      mChannel.update = std::move(mUpdate);
    }
    mChannel.update->id = ++mLastSentId;
    mChannel.guiSleeping = true;
    mUpdate = {};
  }
  lock.unlock();

  if (post) {
    mChannel.updatePosted();
  }
  if (apply) {
    applySettings();
  }
}

void GPUDisplayGUI::applySettings()
{
  if (mState.guiScale != mScale) { // Changed by the display (Reset GUI button) or from a previous GUI session
    setScale(mState.guiScale);
  }
  for (uint32_t id = 0; id < mElementSetters.size() && id < mState.values.size(); id++) {
    if (mElementSetters[id]) {
      mElementSetters[id](mState.values[id]);
    }
  }
}

void GPUDisplayGUI::setValue(uint32_t id, GPUDisplayGUIValue value) { mergeValue(mUpdate.values, id, std::move(value)); }

void GPUDisplayGUI::buildElements(const std::vector<GPUDisplayGUIChannel::Frame>& layout)
{
  using Type = GPUDisplayGUIChannel::ElementType;
  static constexpr int32_t SLIDER_STEPS = 1000; // Resolution of sliders for floating point options
  auto* frames = new QWidget;                   // Frames with their natural width, next to each other while they fit into the window
  auto* framesLayout = new FlowLayout(frames, 1, true);
  mFramesLayout = framesLayout;
  framesLayout->setContentsMargins(0, 0, 0, 0);
  mUi->mainLayout->insertWidget(mUi->mainLayout->count() - 1, frames); // Before the vertical spacer
  for (const auto& frame : layout) {
    auto* box = new QGroupBox(QString::fromStdString(frame.description));
    if (frame.description.empty()) {
      box->setStyleSheet("QGroupBox { margin-top: 0; }"); // No space for a title
    }
    const bool horizontal = frame.horizontal;
    QLayout* boxLayout = horizontal ? (QLayout*)new FlowLayout(box, 3, false) : (QLayout*)new QFormLayout(box); // More space between the elements than between the rows
    for (const auto& el : frame.elements) {
      const uint32_t id = el.id;
      QWidget* widget = nullptr;
      std::function<void(const GPUDisplayGUIValue&)> setter; // Setting a widget programmatically is not a user change, signals emitted by it are blocked
      if (el.type == Type::CHECKBOX) {
        auto* w = new QCheckBox(QString::fromStdString(el.name)); // Own text instead of a label
        connect(w, &QCheckBox::toggled, this, [this, id](bool v) { setValue(id, (int64_t)v); });
        setter = [w](const GPUDisplayGUIValue& v) { const QSignalBlocker blocker(w); w->setChecked(toDouble(v) != 0.); };
        widget = w;
      } else if (el.type == Type::TEXTBOX) {
        auto* w = new QLineEdit;
        if (!el.hasRange) { // With range, any text is accepted and converted / limited
          w->setValidator(el.isInteger ? (QValidator*)new QIntValidator(w) : (QValidator*)new QDoubleValidator(w));
        }
        connect(w, &QLineEdit::editingFinished, this, [this, w, id, isInteger = el.isInteger, hasRange = el.hasRange, min = el.min, max = el.max]() {
          bool ok = false;
          GPUDisplayGUIValue v = isInteger ? GPUDisplayGUIValue((int64_t)w->text().toLongLong(&ok)) : GPUDisplayGUIValue(w->text().toDouble(&ok));
          if (hasRange) {
            const double clamped = std::clamp(ok ? toDouble(v) : 0., min, max);
            v = isInteger ? GPUDisplayGUIValue((int64_t)std::llround(clamped)) : GPUDisplayGUIValue(clamped);
            w->setText(QString::number(clamped)); // Show the value actually set, the display state does not overwrite the focused field
          } else if (!ok) {
            return;
          }
          setValue(id, v);
        });
        setter = [w](const GPUDisplayGUIValue& v) {
          if (!w->hasFocus()) { // Do not overwrite while the user edits
            w->setText(std::visit([](const auto& x) {
              if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, std::string>) {
                return QString::fromStdString(x);
              } else {
                return QString::number(x);
              }
            },
                                  v));
          }
        };
        widget = w;
      } else if (el.type == Type::SLIDER) {
        widget = new QWidget;
        auto* row = new QHBoxLayout(widget);
        row->setContentsMargins(0, 0, 0, 0);
        auto* slider = new QSlider(Qt::Horizontal);
        auto* label = new QLabel;
        if (el.isInteger || el.choices.size()) { // Length by number of steps, about twice a checkbox indicator per step
          const int steps = el.choices.size() ? el.choices.size() - 1 : std::lround(el.max - el.min);
          mScaledWidths.push_back({slider, steps * 2 * style()->pixelMetric(QStyle::PM_IndicatorWidth), true});
        } else {
          mScaledWidths.push_back({slider, 150, false});
        }
        mScaledWidths.push_back({label, 50, false});
        row->addWidget(slider);
        row->addWidget(label);
        if (el.choices.size()) { // Positions are the indices of the choices
          slider->setRange(0, el.choices.size() - 1);
          connect(slider, &QSlider::valueChanged, this, [this, label, id, choices = el.choices](int32_t pos) {
            label->setText(QString::fromStdString(choices[pos].first));
            setValue(id, choices[pos].second);
          });
          setter = [slider, label, choices = el.choices](const GPUDisplayGUIValue& value) {
            uint32_t best = 0; // Nearest choice for values not in the list
            for (uint32_t i = 1; i < choices.size(); i++) {
              if (std::abs(toDouble(choices[i].second) - toDouble(value)) < std::abs(toDouble(choices[best].second) - toDouble(value))) {
                best = i;
              }
            }
            const QSignalBlocker blocker(slider);
            slider->setValue(best);
            label->setText(sameValue(choices[best].second, value) ? QString::fromStdString(choices[best].first) : QString::number(toDouble(value)));
          };
        } else {
          const double min = el.min, step = el.isInteger ? 1. : (el.max - el.min) / SLIDER_STEPS;
          slider->setRange(0, el.isInteger ? std::lround(el.max - el.min) : SLIDER_STEPS);
          connect(slider, &QSlider::valueChanged, this, [this, label, id, min, step, isInteger = el.isInteger](int32_t pos) {
            const double v = min + pos * step;
            label->setNum(v);
            setValue(id, isInteger ? GPUDisplayGUIValue((int64_t)std::llround(v)) : GPUDisplayGUIValue(v));
          });
          setter = [slider, label, min, step](const GPUDisplayGUIValue& value) {
            const double v = toDouble(value);
            const QSignalBlocker blocker(slider);
            slider->setValue(std::lround((v - min) / step));
            label->setNum(v);
          };
        }
      } else if (el.type == Type::RADIO_BUTTONS) {
        widget = new QWidget;
        auto* row = new QHBoxLayout(widget);
        row->setContentsMargins(0, 0, 0, 0);
        auto* group = new QButtonGroup(widget);
        for (uint32_t i = 0; i < el.choices.size(); i++) {
          auto* button = new QRadioButton(QString::fromStdString(el.choices[i].first));
          group->addButton(button, i);
          row->addWidget(button);
        }
        connect(group, &QButtonGroup::idClicked, this, [this, id, choices = el.choices](int32_t i) { setValue(id, choices[i].second); }); // idClicked only on user clicks
        setter = [group, choices = el.choices](const GPUDisplayGUIValue& v) {
          for (uint32_t i = 0; i < choices.size(); i++) {
            if (sameValue(choices[i].second, v)) {
              group->button(i)->setChecked(true);
            }
          }
        };
      } else if (el.type == Type::BUTTON) { // No value from the display
        auto* w = new QPushButton(QString::fromStdString(el.name));
        connect(w, &QPushButton::clicked, this, [this, id]() { setValue(id, (int64_t)1); });
        widget = w;
      } else if (el.type == Type::DROPDOWN) {
        auto* w = new QComboBox;
        for (const auto& choice : el.choices) {
          w->addItem(QString::fromStdString(choice.first));
        }
        connect(w, &QComboBox::currentIndexChanged, this, [this, id, choices = el.choices](int32_t i) {
          if (i >= 0) {
            setValue(id, choices[i].second);
          }
        });
        setter = [w, choices = el.choices](const GPUDisplayGUIValue& v) {
          const QSignalBlocker blocker(w);
          for (uint32_t i = 0; i < choices.size(); i++) {
            if (sameValue(choices[i].second, v)) {
              w->setCurrentIndex(i);
            }
          }
        };
        widget = w;
      }
      QLabel* name = el.type == Type::CHECKBOX || el.type == Type::BUTTON ? nullptr : new QLabel(QString::fromStdString(el.name)); // Checkboxes and buttons show their own text
      if (!horizontal) {
        name ? ((QFormLayout*)boxLayout)->addRow(name, widget) : ((QFormLayout*)boxLayout)->addRow(widget);
      } else if (name) { // Label and element as one item of the flow layout
        auto* item = new QWidget;
        auto* row = new QHBoxLayout(item);
        row->setContentsMargins(0, 0, 0, 0);
        row->addWidget(name);
        row->addWidget(widget);
        boxLayout->addWidget(item);
      } else {
        boxLayout->addWidget(widget);
      }
      if (mElementSetters.size() <= id) {
        mElementSetters.resize(id + 1);
      }
      mElementSetters[id] = std::move(setter);
    }
    framesLayout->addWidget(box);
    if (std::any_of(frame.elements.begin(), frame.elements.end(), [](const auto& el) { return el.type == Type::SLIDER && !el.isInteger && el.choices.empty(); })) {
      mSliderFrames.emplace_back(box); // Preferred width set in setScale(), sliders range to the frame border, frame limited to the window width by the layout
    }
  }
}

void GPUDisplayGUI::setScale(float scale)
{
  mScale = std::clamp(scale, 0.25f, 4.f);
  for (const auto& w : mScaledWidths) {
    if (w.fixed) {
      w.widget->setFixedWidth(std::lround(w.width * mScale));
    } else {
      w.widget->setMinimumWidth(std::lround(w.width * mScale));
    }
  }
  for (auto* box : mSliderFrames) {
    static_cast<FlowLayout*>(mFramesLayout)->setPreferredWidth(box, std::lround(screen()->availableGeometry().width() / 3 * mScale));
  }
  if (mFramesLayout) {
    mFramesLayout->invalidate();
  }
  const int indicator = std::lround(style()->pixelMetric(QStyle::PM_IndicatorWidth) * mScale);
  setStyleSheet(QString("QGroupBox { border: 1px solid palette(mid); border-radius: 3px; margin-top: 0.6em; padding-top: 0.3em; }"
                        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }" // Title on the frame border instead of in an extra line above
                        "QCheckBox::indicator, QRadioButton::indicator { width: %1px; height: %1px; }"
                        "QWidget { font-size: %2pt; }") // Style sheet, since setFont() does not override the fonts the platform theme sets for some classes (QLabel, QPushButton), sizes derived from the text follow
                  .arg(indicator)
                  .arg(mBaseFontSize * mScale, 0, 'f', 1));
}

bool GPUDisplayGUI::eventFilter(QObject* watched, QEvent* event)
{
  const auto* w = qobject_cast<QWidget*>(watched);
  if (!w || w->window() != this) {
    return false;
  }
  if (event->type() == QEvent::Wheel && (static_cast<QWheelEvent*>(event)->modifiers() & Qt::ControlModifier)) {
    const int delta = static_cast<QWheelEvent*>(event)->angleDelta().y(); // 120 per notch
    if (delta) {
      setScale(mScale * std::pow(1.1f, delta / 120.f));
      mUpdate.guiScale = mScale;
    }
    return true;
  }
  if (event->type() == QEvent::KeyPress && static_cast<QKeyEvent*>(event)->key() == Qt::Key_0 && (static_cast<QKeyEvent*>(event)->modifiers() & Qt::ControlModifier)) {
    setScale(1.f);
    mUpdate.guiScale = mScale;
    return true;
  }
  return false;
}

void GPUDisplayGUI::bringToFront()
{
  setWindowState((windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
  show();
  raise();
  activateWindow();
}

void GPUDisplayGUI::keyPressEvent(QKeyEvent* event)
{
  const int32_t key = translateKey(event);
  if (key < 0) {
    QMainWindow::keyPressEvent(event);
    return;
  }
  mUpdate.keys.emplace_back(key);
}

int32_t GPUDisplayGUI::translateKey(const QKeyEvent* event)
{
  switch (event->key()) {
    case Qt::Key_Up:
      return GPUDisplayFrontend::KEY_UP;
    case Qt::Key_Down:
      return GPUDisplayFrontend::KEY_DOWN;
    case Qt::Key_Left:
      return GPUDisplayFrontend::KEY_LEFT;
    case Qt::Key_Right:
      return GPUDisplayFrontend::KEY_RIGHT;
    case Qt::Key_PageUp:
      return GPUDisplayFrontend::KEY_PAGEUP;
    case Qt::Key_PageDown:
      return GPUDisplayFrontend::KEY_PAGEDOWN;
    case Qt::Key_Home:
      return GPUDisplayFrontend::KEY_HOME;
    case Qt::Key_End:
      return GPUDisplayFrontend::KEY_END;
    case Qt::Key_Insert:
      return GPUDisplayFrontend::KEY_INSERT;
    case Qt::Key_Escape:
      return GPUDisplayFrontend::KEY_ESCAPE;
    case Qt::Key_Return:
    case Qt::Key_Enter:
      return GPUDisplayFrontend::KEY_ENTER;
    default:
      break;
  }
  if (event->key() >= Qt::Key_F1 && event->key() <= Qt::Key_F12) {
    static_assert(GPUDisplayFrontend::KEY_F12 - GPUDisplayFrontend::KEY_F1 == Qt::Key_F12 - Qt::Key_F1);
    return GPUDisplayFrontend::KEY_F1 + (event->key() - Qt::Key_F1);
  }
  const QString text = event->text();
  if (text.size() == 1 && text[0].unicode() >= 32 && text[0].unicode() < 127) { // Printable ASCII, with shift applied
    return text[0].unicode();
  }
  return -1;
}
