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
#include "GPUDisplayFrontendKeyVars.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHash>
#include <QHBoxLayout>
#include <QApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPlainTextEdit>
#include <QProxyStyle>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QScreen>
#include <QSignalBlocker>
#include <QSlider>
#include <QStyle>
#include <QStyleHints>
#include <QStatusBar>
#include <QStyleOptionSizeGrip>
#include <QStylePainter>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace o2::gpu;

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
          item->setGeometry(mEqualRowHeight ? QRect(geometry.x(), geometry.y(), geometry.width(), lineHeight) : geometry.translated(0, (lineHeight - geometry.height()) / 2)); // Otherwise vertically centered in the row
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
class GUIStyle : public QProxyStyle // Tooltips only after hovering for 1.5 seconds, also directly after another tooltip
{
 public:
  int styleHint(StyleHint hint, const QStyleOption* option, const QWidget* widget, QStyleHintReturn* returnData) const override
  {
    return hint == SH_ToolTip_WakeUpDelay ? 1500 : (hint == SH_ToolTip_FallAsleepDelay ? 0 : QProxyStyle::styleHint(hint, option, widget, returnData));
  }
};

class ResizeGrip : public QWidget // Size grip in the bottom right corner of its parent, reports the new size while dragged (QSizeGrip resizes only windows)
{
 public:
  ResizeGrip(QWidget* parent, std::function<void(QSize)> resize) : QWidget(parent), mResize(std::move(resize))
  {
    setCursor(Qt::SizeFDiagCursor);
    setFixedSize(16, 16);
  }

 protected:
  void paintEvent(QPaintEvent*) override
  {
    QStylePainter painter(this);
    QStyleOptionSizeGrip opt;
    opt.initFrom(this);
    opt.corner = Qt::BottomRightCorner;
    painter.drawControl(QStyle::CE_SizeGrip, opt);
  }
  void mousePressEvent(QMouseEvent* e) override
  {
    mStartPos = e->globalPosition().toPoint();
    mStartSize = parentWidget()->size();
  }
  void mouseMoveEvent(QMouseEvent* e) override
  {
    if (e->buttons() & Qt::LeftButton) {
      const QPoint d = e->globalPosition().toPoint() - mStartPos;
      mResize(mStartSize + QSize(d.x(), d.y()));
    }
  }

 private:
  std::function<void(QSize)> mResize;
  QPoint mStartPos;
  QSize mStartSize;
};

class TextEdit : public QPlainTextEdit // Text edit with a resize grip, reports a changed text when it loses the focus
{
 public:
  TextEdit(uint32_t lines, std::function<void(const QString&)> changed) : mChanged(std::move(changed)), mGrip(new ResizeGrip(this, [this](QSize size) {
                                                                            mUnscaledSize = size.expandedTo(QSize(60, 30) * mScale) / mScale;
                                                                            setScale(mScale);
                                                                          }))
  {
    const int margins = 2 * (frameWidth() + std::lround(document()->documentMargin())) + 6; // Few more pixels, otherwise a scroll bar appears
    mUnscaledSize = QSize(400, std::max<uint32_t>(lines, 1) * fontMetrics().lineSpacing() + margins);
  }
  QSize sizeHint() const override { return mPreferredSize; }
  void setScale(float scale) // Fixed height, but narrower than preferred if the window is too small, so that it does not determine the width of the other elements
  {
    mScale = scale;
    mPreferredSize = mUnscaledSize * scale;
    setFixedHeight(mPreferredSize.height());
    setMinimumWidth(std::min(mPreferredSize.width(), 100));
    setMaximumWidth(mPreferredSize.width());
    updateGeometry();
  }
  void showText(const QString& text) // Text from the display, not shown while the user edits
  {
    mCurrent = text;
    if (!hasFocus() && toPlainText() != text) {
      setPlainText(text);
      document()->setModified(false);
    }
  }

 protected:
  void focusOutEvent(QFocusEvent* e) override
  {
    QPlainTextEdit::focusOutEvent(e);
    if (document()->isModified()) {
      document()->setModified(false);
      mCurrent = toPlainText();
      mChanged(mCurrent);
    } else {
      showText(mCurrent);
    }
  }
  void resizeEvent(QResizeEvent* e) override
  {
    QPlainTextEdit::resizeEvent(e);
    mGrip->move(width() - mGrip->width(), height() - mGrip->height());
    mGrip->raise();
  }

 private:
  std::function<void(const QString&)> mChanged;
  QString mCurrent;
  QSize mUnscaledSize; // At GUI scale 1, changed by the resize grip
  QSize mPreferredSize;
  float mScale = 1.f;
  ResizeGrip* mGrip;
};

struct ValueEdit { // Text box for a value from the display, copied into the lambdas of the element
  QLineEdit* edit = new QLineEdit;
  std::shared_ptr<QString> current = std::make_shared<QString>(); // Restored when the text box is left without a valid new value

  void show(const QString& text) const // Value from the display, not shown while the user edits
  {
    *current = text;
    if (!edit->hasFocus()) {
      edit->setText(text);
    }
  }
  void commit(const QString& text) const // New value entered by the user, possibly corrected
  {
    *current = text;
    edit->setText(text);
  }
  std::optional<QString> entered() const // Text if changed by the user, otherwise restores the current value
  {
    if (edit->isModified()) {
      return edit->text();
    }
    edit->setText(*current);
    return std::nullopt;
  }
  std::optional<double> number() const // Number if entered by the user, otherwise restores the current value
  {
    bool ok = false;
    const double v = edit->text().toDouble(&ok);
    if (edit->isModified() && ok && std::isfinite(v)) {
      return v;
    }
    edit->setText(*current);
    return std::nullopt;
  }
};

class MultiSlider : public QWidget // Handles on one range [0, value.max] with ticks below: dragged with the mouse (multiples of step, never onto another handle), a click without moving (de)selects
{
 public:
  explicit MultiSlider(double step) : mStep(step)
  {
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setFocusPolicy(Qt::ClickFocus); // A click finishes the editing in the text box first, for the previously focused handle
  }
  std::function<void()> changed;      // Positions or selection changed by the user, value in the order received from the display
  std::function<void()> focusChanged; // Other focused handle or its position, for the text box
  GPUDisplayMultiSliderValue value;   // Positions in the order of the display, selected handle
  int32_t focused = -1;               // Last clicked / dragged handle, shown in the text box

  void setFromDisplay(const GPUDisplayMultiSliderValue& v)
  {
    value = v;
    mDragging = -1; // A drag is cancelled by changes from the display
    focused = focused < (int32_t)value.values.size() ? focused : -1;
    update();
    focusChanged();
  }
  void setPosition(int32_t i, double v) // From the text box, rounded to the step, ignored if the position does not change or another handle is there
  {
    v = snap(v);
    if (isFree(v, -1)) { // Also the own position
      value.values[i] = v;
      sendAndNormalize();
    }
  }
  QSize sizeHint() const override { return {300, handleHeight() + 6 + fontMetrics().height() + 8}; }
  QSize minimumSizeHint() const override { return {100, sizeHint().height()}; }

 protected:
  void paintEvent(QPaintEvent*) override
  {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QPalette& pal = palette();
    const int32_t trackY = handleHeight() / 2 + 1, tickY = handleHeight() + 3;
    p.setPen(QPen(pal.color(QPalette::Mid), 2));
    p.drawLine(margin(), trackY, width() - margin(), trackY);
    if (value.progress && value.values.size()) { // Up to the progress of the last position, e.g. the running animation
      p.setPen(QPen(pal.color(QPalette::Highlight), 4));
      p.drawLine(margin(), trackY, toX(value.progress / 65535. * value.values.back()), trackY);
    }
    p.setPen(pal.color(QPalette::WindowText));
    const double tick = tickStep();
    for (int32_t k = 0; k * tick <= maxValue() * (1 + 1e-9); k++) {
      const int32_t x = toX(k * tick);
      p.drawLine(x, tickY, x, tickY + 4);
      p.drawText(QRect(x - 50, tickY + 5, 100, fontMetrics().height()), Qt::AlignHCenter | Qt::AlignTop, QString::number(k * tick));
    }
    for (uint32_t i = 0; i < value.values.size(); i++) {
      const QRect r(toX(value.values[i]) - handleWidth() / 2, 1, handleWidth(), handleHeight());
      p.setBrush((int32_t)i == value.selected ? pal.color(QPalette::Highlight) : pal.color(QPalette::Button));
      p.setPen(QPen((int32_t)i == focused ? pal.color(QPalette::WindowText) : pal.color(QPalette::Mid), (int32_t)i == focused ? 2 : 1));
      p.drawRoundedRect(r, 2, 2);
    }
  }
  void mousePressEvent(QMouseEvent* e) override
  {
    if (e->button() != Qt::LeftButton) {
      return;
    }
    int32_t best = -1, bestDistance = handleWidth();
    for (uint32_t i = 0; i < value.values.size(); i++) { // Nearest handle under the mouse
      if (const int32_t d = std::abs(toX(value.values[i]) - (int32_t)e->position().x()); d <= bestDistance) {
        best = i;
        bestDistance = d;
      }
    }
    if (best >= 0) {
      mDragging = focused = best;
      mMoved = false;
      update();
      focusChanged();
    }
  }
  void mouseMoveEvent(QMouseEvent* e) override
  {
    if (mDragging < 0) {
      return;
    }
    const float v = snap(toValue(e->position().x()));           // Compared to the stored float
    if (isFree(v, mDragging) && v != value.values[mDragging]) { // Never onto another handle, it can be passed
      value.values[mDragging] = v;
      mMoved = true;
      update();
      focusChanged();
    }
  }
  void mouseReleaseEvent(QMouseEvent* e) override
  {
    if (e->button() != Qt::LeftButton || mDragging < 0) {
      return;
    }
    if (!mMoved) {
      value.selected = value.selected == mDragging ? -1 : mDragging;
    }
    mDragging = -1;
    sendAndNormalize();
  }

 private:
  void sendAndNormalize() // Sends the value in the order of the display, then sorts it as the display does (first position 0), until its reply arrives
  {
    changed();
    std::vector<uint32_t> order(value.values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [this](uint32_t a, uint32_t b) { return value.values[a] < value.values[b]; });
    GPUDisplayMultiSliderValue sorted = value;
    int32_t newFocused = -1;
    for (uint32_t i = 0; i < order.size(); i++) {
      sorted.values[i] = value.values[order[i]] - value.values[order[0]];
      sorted.selected = (int32_t)order[i] == value.selected ? i : sorted.selected;
      newFocused = (int32_t)order[i] == focused ? i : newFocused;
    }
    value = std::move(sorted);
    focused = newFocused;
    update();
    focusChanged();
  }
  double maxValue() const { return std::max<double>(value.max, mStep); }
  double snap(double v) const { return std::clamp(std::round(v / mStep) * mStep, 0., maxValue()); } // Positions are multiples of the step
  bool isFree(double v, int32_t except) const                                                       // No handle other than except at v (positions are snapped: closer than half a step is the same position)
  {
    return std::none_of(value.values.begin(), value.values.end(), [&](const float& x) { return &x - value.values.data() != except && std::abs(x - v) < mStep / 2; });
  }
  int32_t handleHeight() const { return fontMetrics().height(); }
  int32_t handleWidth() const { return std::max(6, fontMetrics().height() / 2); }
  int32_t margin() const { return std::max(handleWidth() / 2 + 1, fontMetrics().horizontalAdvance("00.00") / 2); } // Room for the handles and tick labels at the ends
  int32_t toX(double v) const { return margin() + std::lround(v / maxValue() * (width() - 2 * margin())); }
  double toValue(double x) const { return (x - margin()) / std::max(1, width() - 2 * margin()) * maxValue(); }
  double tickStep() const // 1, 2, 5 times a power of 10, labels not overlapping
  {
    const double minStep = maxValue() * 2 * fontMetrics().horizontalAdvance("00.00") / std::max(1, width() - 2 * margin());
    const double base = std::pow(10., std::floor(std::log10(std::max(minStep, 1e-9))));
    for (double m : {1., 2., 5.}) {
      if (m * base >= minStep) {
        return std::max(m * base, mStep);
      }
    }
    return std::max(10 * base, mStep);
  }

  double mStep;
  int32_t mDragging = -1;
  bool mMoved = false;
};

struct SliderRange {                           // Slider positions of a range of values
  static constexpr int32_t FLOAT_STEPS = 1000; // Resolution of sliders for floating point options
  double min, step;
  int32_t steps;
  SliderRange(double minValue, double maxValue, bool isInteger, double fixedStep = 0) : min(minValue), step(fixedStep > 0 ? fixedStep : (isInteger ? 1. : (maxValue - minValue) / FLOAT_STEPS)), steps(fixedStep > 0 || isInteger ? std::max<int32_t>(std::lround((maxValue - minValue) / step), 0) : FLOAT_STEPS) {} // fixedStep: Element::step
  double value(int32_t position) const { return min + position * step; }
  int32_t position(double v) const { return step ? std::lround((v - min) / step) : 0; }
};
} // namespace

static float clampScale(float scale) { return std::clamp(scale, 0.25f, 4.f); }

static bool fixedLengthSlider(const GPUDisplayGUIChannel::Element& el) // Discrete / integer sliders with few steps, others stretch to the frame border
{
  static constexpr double MAX_FIXED_LENGTH_STEPS = 16;
  return !el.dynamicOptions && (el.choices.size() || (el.isInteger && el.max - el.min <= MAX_FIXED_LENGTH_STEPS)); // Dynamic range: number of steps may change
}

static double toDouble(const GPUDisplayGUIValue& v)
{
  return std::visit([](const auto& x) -> double {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, std::string>) {
      return QString::fromStdString(x).toDouble();
    } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, GPUDisplayMultiSliderValue>) {
      return 0.;
    } else {
      return x;
    }
  },
                    v);
}

static QString toQString(const GPUDisplayGUIValue& v)
{
  return std::visit([](const auto& x) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, std::string>) {
      return QString::fromStdString(x);
    } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, GPUDisplayMultiSliderValue>) {
      return QString();
    } else {
      return QString::number(x);
    }
  },
                    v);
}

static GPUDisplayGUIValue toValue(double v, bool isInteger) { return isInteger ? GPUDisplayGUIValue((int64_t)std::llround(v)) : GPUDisplayGUIValue(v); }

static uint32_t nearestChoice(const std::vector<std::pair<std::string, GPUDisplayGUIValue>>& choices, double value) // For values not in the list
{
  uint32_t best = 0;
  for (uint32_t i = 1; i < choices.size(); i++) {
    if (std::abs(toDouble(choices[i].second) - value) < std::abs(toDouble(choices[best].second) - value)) {
      best = i;
    }
  }
  return best;
}

static bool sameValue(const GPUDisplayGUIValue& a, const GPUDisplayGUIValue& b) // Numbers compared by value, independent of int64_t / double
{
  const bool aNumber = std::holds_alternative<int64_t>(a) || std::holds_alternative<double>(a), bNumber = std::holds_alternative<int64_t>(b) || std::holds_alternative<double>(b);
  return aNumber && bNumber ? toDouble(a) == toDouble(b) : a == b;
}

GPUDisplayGUI::GPUDisplayGUI(GPUDisplayGUIChannel& channel, QWidget* parent) : QMainWindow(parent), mChannel(channel)
{
  qApp->setStyle(new GUIStyle); // Owned by the application
  resize(1370, 1210);
  setMinimumSize(500, 500);
  mStatusLabel = new QLabel;
  mStatusLabel->setWordWrap(true);
  mStatusLabel->hide();
  statusBar()->addWidget(mStatusLabel, 1);
  mDarkMode = darkPalette();                                                                                                                // System scheme, until the display tells otherwise
  connect(QGuiApplication::styleHints(), &QStyleHints::colorSchemeChanged, this, &GPUDisplayGUI::colorSchemeChanged, Qt::QueuedConnection); // Queued: after the palette was updated
  mInfoTextTimer.setSingleShot(true);
  connect(&mInfoTextTimer, &QTimer::timeout, this, &GPUDisplayGUI::updateInfoLabels);
  std::vector<GPUDisplayGUIChannel::Frame> layout;
  {
    std::lock_guard lock(mChannel.mutex);
    setWindowTitle(QString::fromStdString(mChannel.windowName + " GUI"));
    layout = mChannel.layout;
  }
  mBaseFontSize = font().pointSizeF() > 0 ? font().pointSizeF() : 10.f; // Fonts with pixel size have no point size
  auto* content = new QWidget;
  auto* contentLayout = new QVBoxLayout(content);
  contentLayout->addWidget(buildElements(layout));
  contentLayout->addStretch();
  auto* scroll = new QScrollArea; // Vertical scroll bar if the frames do not fit into the window, they wrap to the window width
  scroll->setWidgetResizable(true);
  scroll->setFrameShape(QFrame::NoFrame);
  scroll->setWidget(content);
  setCentralWidget(scroll);
  colorSchemeChanged();           // Style sheets, with the GUI scale, which comes from the display with the first state
  qApp->installEventFilter(this); // Sliders / dropdowns would consume Ctrl + mouse wheel themselves
  connect(&mTimer, &QTimer::timeout, this, &GPUDisplayGUI::timerTick);
  mTimer.start(std::chrono::milliseconds(10));
}

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
  std::vector<std::pair<uint32_t, GPUDisplayGUIChannel::Options>> options; // Read while the display waits, applied to the widgets after unlocking
  if (mChannel.displayWaitingForOptions && mChannel.state.optionsRevision != mOptionsRevision) {
    for (uint32_t id = 0; id < mElements.size(); id++) {
      if (mElements[id].getOptions) {
        options.emplace_back(id, mElements[id].getOptions());
      }
    }
    mOptionsRevision = mChannel.optionsReadRevision = mChannel.state.optionsRevision;
    mChannel.optionsRead.notify_all();
  }
  const bool apply = newState && mUpdate.empty() && mState.appliedUpdate == mLastSentId; // State contains all our updates, and we have no unsent changes
  const bool post = !mUpdate.empty() && mLastSentId - mLastAppliedId < mChannel.maxUpdatesAhead;
  if (post) {
    if (mChannel.update) { // Previous update not yet picked up by the display, merge
      mChannel.update->merge(mUpdate);
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
  if (newState) {
    if (mState.cfg.mCfgO.guiDarkMode < 0) {
      mUpdate.guiDarkMode = mDarkMode; // Report the system scheme, also shows up in the DarkMode checkbox
    } else if ((mState.cfg.mCfgO.guiDarkMode != 0) != mDarkMode) {
      setDarkMode(mState.cfg.mCfgO.guiDarkMode != 0);
    }
    if (mState.infoTextCounter != mInfoTextCounter) {
      mInfoTextCounter = mState.infoTextCounter;
      mInfoTextTimer.start(mChannel.infoTextTime);
    }
    updateInfoLabels();
  }
  for (const auto& [id, o] : options) { // New range / choices, then the current value again
    mElements[id].setOptions(o);
    mElements[id].applied.reset();
    applyElement(mElements[id]);
  }
  if (apply) {
    applySettings();
  }
}

bool GPUDisplayGUI::darkPalette() const { return palette().window().color().lightness() < 128; }

void GPUDisplayGUI::setDarkMode(bool dark)
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 8, 0)
  QGuiApplication::styleHints()->setColorScheme(dark ? Qt::ColorScheme::Dark : Qt::ColorScheme::Light); // No effect if the platform theme does not support it
#endif
  mDarkMode = dark;
}

void GPUDisplayGUI::colorSchemeChanged()
{
  setStyleSheet(QString()); // Otherwise the widgets keep their old palette
  setScale(mScale);         // Sets the style sheet again
  updateInfoLabels();
}

void GPUDisplayGUI::resizeEvent(QResizeEvent* event)
{
  QMainWindow::resizeEvent(event);
  updateInfoLabels(); // Shows the GUI size
}

void GPUDisplayGUI::fitStatusLabel()
{
  mStatusLabel->setMinimumHeight(0); // heightForWidth of QLabel does not return less than the minimum height
  mStatusLabel->setFixedHeight(mStatusLabel->heightForWidth(mStatusLabel->width()));
}

void GPUDisplayGUI::updateInfoLabels()
{
  const QString status = QString::fromStdString(mState.infoStatus) + QString(" GUI %1x%2").arg(width()).arg(height());
  if (mInfoTextTimer.isActive() && !mState.infoText.empty()) { // Info message for some seconds, in bold and color, in the same line
    const char* color = darkPalette() ? "lightblue" : "darkblue";
    mStatusLabel->setTextFormat(Qt::RichText);
    mStatusLabel->setText("<span style=\"white-space: pre-wrap;\">" + status.toHtmlEscaped() + QString("</span> - <span style=\"font-weight: bold; color: %1;\">%2</span>").arg(color, QString::fromStdString(mState.infoText).toHtmlEscaped()));
  } else {
    mStatusLabel->setTextFormat(Qt::PlainText);
    mStatusLabel->setText(status);
  }
  mStatusLabel->setVisible(!mState.infoStatus.empty());
  fitStatusLabel();
}

void GPUDisplayGUI::applySettings()
{
  if (clampScale(mState.cfg.mCfgO.guiScale) != mScale) { // Changed by the display (Reset GUI button, console) or from a previous GUI session, clamped as by setScale
    setScale(mState.cfg.mCfgO.guiScale);
  }
  for (auto& element : mElements) {
    applyElement(element);
  }
}

void GPUDisplayGUI::applyElement(GUIElement& element)
{
  if (!element.set) {
    return;
  }
  GPUDisplayGUIValue value = element.get(mState.cfg);
  if (element.applied != value) {
    element.set(value);
    element.applied = std::move(value);
  }
}

void GPUDisplayGUI::setValue(uint32_t id, GPUDisplayGUIValue value)
{
  mUpdate.setValue(id, std::move(value));
  mElements[id].applied.reset();
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createCheckbox(const GPUDisplayGUIChannel::Element& el)
{
  auto* w = new QCheckBox(QString::fromStdString(el.name)); // Own text instead of a label
  connect(w, &QCheckBox::toggled, this, [this, id = el.id](bool v) { setValue(id, (int64_t)v); });
  return {w, [w](const GPUDisplayGUIValue& v) { const QSignalBlocker blocker(w); w->setChecked(toDouble(v) != 0.); }, {}};
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createTextbox(const GPUDisplayGUIChannel::Element& el)
{
  ValueEdit text;
  connect(text.edit, &QLineEdit::editingFinished, this, [this, text, id = el.id, isInteger = el.isInteger, isString = el.isString, hasRange = el.hasRange, min = el.min, max = el.max]() {
    if (isString) {
      if (const auto str = text.entered()) {
        text.commit(*str);
        setValue(id, str->toStdString());
      }
    } else if (const auto entered = text.number()) {
      const GPUDisplayGUIValue v = toValue(hasRange ? std::clamp(*entered, min, max) : *entered, isInteger);
      text.commit(toQString(v)); // Show the value actually set, the display state does not overwrite the focused field
      setValue(id, v);
    }
  });
  return {text.edit, [text](const GPUDisplayGUIValue& v) { text.show(toQString(v)); }, {}};
}

QHBoxLayout* GPUDisplayGUI::createSliderRow(ElementWidget& retVal, QWidget* slider, QLineEdit* edit, Qt::Alignment editAlignment)
{
  retVal.widget = new QWidget;
  auto* row = new QHBoxLayout(retVal.widget);
  row->setContentsMargins(0, 0, 0, 0);
  mScaleSetters.emplace_back([edit](float scale) { edit->setFixedWidth(std::lround(60 * scale)); });
  row->addWidget(slider);
  row->addWidget(edit, 0, editAlignment);
  return row;
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createSlider(const GPUDisplayGUIChannel::Element& el)
{
  ElementWidget retVal;
  auto* slider = new QSlider(Qt::Horizontal);
  ValueEdit text;
  auto* row = createSliderRow(retVal, slider, text.edit);
  const bool fixedLength = fixedLengthSlider(el);
  const int32_t length = fixedLength ? (el.choices.size() ? el.choices.size() - 1 : std::lround(el.max - el.min)) * 2 * style()->pixelMetric(QStyle::PM_IndicatorWidth) : 150; // Fixed: about twice a checkbox indicator per step
  mScaleSetters.emplace_back([slider, length, fixedLength](float scale) {
    if (fixedLength) {
      slider->setFixedWidth(std::lround(length * scale));
    } else {
      slider->setMinimumWidth(std::lround(length * scale));
    }
  });
  if (fixedLength) {
    row->addStretch(); // Left-aligned, otherwise the free space is distributed around the fixed-size widgets
  }
  if (el.choices.size()) { // Positions are the indices of the choices
    auto choices = std::make_shared<const std::vector<std::pair<std::string, GPUDisplayGUIValue>>>(el.choices);
    slider->setRange(0, choices->size() - 1);
    connect(slider, &QSlider::valueChanged, this, [this, text, id = el.id, choices](int32_t pos) {
      text.show(QString::fromStdString((*choices)[pos].first));
      setValue(id, (*choices)[pos].second);
    });
    connect(text.edit, &QLineEdit::editingFinished, this, [this, text, slider, id = el.id, choices]() {
      if (const auto entered = text.number()) {
        const uint32_t pos = nearestChoice(*choices, *entered);
        text.commit(QString::fromStdString((*choices)[pos].first));
        const QSignalBlocker blocker(slider);
        slider->setValue(pos);
        setValue(id, (*choices)[pos].second);
      }
    });
    retVal.set = [slider, text, choices](const GPUDisplayGUIValue& value) {
      const uint32_t best = nearestChoice(*choices, toDouble(value));
      const QSignalBlocker blocker(slider);
      slider->setValue(best);
      text.show(sameValue((*choices)[best].second, value) ? QString::fromStdString((*choices)[best].first) : toQString(value));
    };
  } else {
    auto range = std::make_shared<SliderRange>(el.min, el.max, el.isInteger, el.step); // Changed by dynamic options
    slider->setRange(0, range->steps);
    connect(slider, &QSlider::valueChanged, this, [this, text, id = el.id, range, isInteger = el.isInteger](int32_t pos) {
      const GPUDisplayGUIValue v = toValue(range->value(pos), isInteger);
      text.show(toQString(v));
      setValue(id, v);
    });
    connect(text.edit, &QLineEdit::editingFinished, this, [this, text, slider, id = el.id, range, isInteger = el.isInteger, step = el.step]() {
      if (const auto entered = text.number()) {
        const double clamped = std::clamp(*entered, range->min, range->value(range->steps));
        const GPUDisplayGUIValue v = toValue(step > 0 ? range->value(range->position(clamped)) : clamped, isInteger); // Rounded to the step if set, otherwise floating point values need not be on a slider step
        text.commit(toQString(v));
        const QSignalBlocker blocker(slider);
        slider->setValue(range->position(toDouble(v)));
        setValue(id, v);
      }
    });
    retVal.set = [slider, text, range](const GPUDisplayGUIValue& value) {
      const QSignalBlocker blocker(slider);
      slider->setValue(range->position(toDouble(value)));
      text.show(toQString(value));
    };
    retVal.setOptions = [slider, range, isInteger = el.isInteger, step = el.step](const GPUDisplayGUIChannel::Options& o) {
      *range = SliderRange(o.min, o.max, isInteger, step);
      const QSignalBlocker blocker(slider);
      slider->setRange(0, range->steps);
    };
  }
  return retVal;
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createMultiSlider(const GPUDisplayGUIChannel::Element& el)
{
  ElementWidget retVal;
  auto* slider = new MultiSlider(el.step);
  ValueEdit text;
  createSliderRow(retVal, slider, text.edit, Qt::AlignTop);
  mScaleSetters.emplace_back([slider](float scale) { slider->setMinimumWidth(std::lround(300 * scale)); });
  slider->changed = [this, slider, id = el.id]() { setValue(id, slider->value); };
  auto focusedText = [slider]() { return slider->focused >= 0 ? QString::number(slider->value.values[slider->focused]) : QString(); };
  slider->focusChanged = [text, focusedText]() { text.show(focusedText()); };
  connect(text.edit, &QLineEdit::editingFinished, this, [slider, text, focusedText]() {
    if (const auto entered = text.number(); entered && slider->focused >= 0) {
      slider->setPosition(slider->focused, *entered);
    }
    text.commit(focusedText()); // The position actually set, also after the normalization
  });
  retVal.set = [slider](const GPUDisplayGUIValue& v) { slider->setFromDisplay(std::get<GPUDisplayMultiSliderValue>(v)); };
  return retVal;
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createRadioButtons(const GPUDisplayGUIChannel::Element& el)
{
  auto* widget = new QWidget;
  auto* row = new QHBoxLayout(widget);
  row->setContentsMargins(0, 0, 0, 0);
  auto* group = new QButtonGroup(widget);
  for (uint32_t i = 0; i < el.choices.size(); i++) {
    auto* button = new QRadioButton(QString::fromStdString(el.choices[i].first));
    group->addButton(button, i);
    row->addWidget(button);
  }
  connect(group, &QButtonGroup::idClicked, this, [this, id = el.id, choices = el.choices](int32_t i) { setValue(id, choices[i].second); }); // idClicked only on user clicks
  return {widget, [group, choices = el.choices](const GPUDisplayGUIValue& v) {
            for (uint32_t i = 0; i < choices.size(); i++) {
              if (sameValue(choices[i].second, v)) {
                group->button(i)->setChecked(true);
              }
            }
          },
          {}};
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createTextEdit(const GPUDisplayGUIChannel::Element& el)
{
  auto* w = new TextEdit(el.lines, [this, id = el.id](const QString& text) { setValue(id, text.toStdString()); });
  mScaleSetters.emplace_back([w](float scale) { w->setScale(scale); });
  return {w, [w](const GPUDisplayGUIValue& v) { w->showText(toQString(v)); }, {}};
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createButton(const GPUDisplayGUIChannel::Element& el) // No value from the display
{
  auto* w = new QPushButton(QString::fromStdString(el.name));
  connect(w, &QPushButton::clicked, this, [this, id = el.id]() { setValue(id, (int64_t)1); });
  return {w, {}, {}};
}

GPUDisplayGUI::ElementWidget GPUDisplayGUI::createDropdown(const GPUDisplayGUIChannel::Element& el)
{
  auto* w = new QComboBox;
  auto choices = std::make_shared<std::vector<std::pair<std::string, GPUDisplayGUIValue>>>(el.choices); // Changed by dynamic options
  for (const auto& choice : *choices) {
    w->addItem(QString::fromStdString(choice.first));
  }
  connect(w, &QComboBox::currentIndexChanged, this, [this, id = el.id, choices](int32_t i) {
    if (i >= 0 && i < (int32_t)choices->size()) {
      setValue(id, (*choices)[i].second);
    }
  });
  return {w, [w, choices](const GPUDisplayGUIValue& v) {
            const QSignalBlocker blocker(w);
            for (uint32_t i = 0; i < choices->size(); i++) {
              if (sameValue((*choices)[i].second, v)) {
                w->setCurrentIndex(i);
              }
            } },
          [w, choices](const GPUDisplayGUIChannel::Options& o) {
            *choices = o.choices;
            const QSignalBlocker blocker(w);
            w->clear();
            for (const auto& choice : *choices) {
              w->addItem(QString::fromStdString(choice.first));
            }
          }};
}

QWidget* GPUDisplayGUI::buildElements(const std::vector<GPUDisplayGUIChannel::Frame>& layout)
{
  using Type = GPUDisplayGUIChannel::ElementType;
  auto* frames = new QWidget; // Frames with their natural width, next to each other while they fit into the window
  auto* framesLayout = new FlowLayout(frames, 1, true);
  framesLayout->setContentsMargins(0, 0, 0, 0);
  for (const auto& frame : layout) {
    auto* box = new QGroupBox(QString::fromStdString(frame.description));
    if (frame.description.empty()) {
      box->setStyleSheet("QGroupBox { margin-top: 0; }"); // No space for a title
    }
    auto* boxLayout = new QVBoxLayout(box);
    QLayout* section = nullptr; // Consecutive elements with the same alignment
    bool sectionHorizontal = false;
    for (const auto& el : frame.elements) {
      ElementWidget created;
      switch (el.type) {
        case Type::CHECKBOX:
          created = createCheckbox(el);
          break;
        case Type::TEXTBOX:
          created = createTextbox(el);
          break;
        case Type::SLIDER:
          created = createSlider(el);
          break;
        case Type::RADIO_BUTTONS:
          created = createRadioButtons(el);
          break;
        case Type::TEXTEDIT:
          created = createTextEdit(el);
          break;
        case Type::BUTTON:
          created = createButton(el);
          break;
        case Type::DROPDOWN:
          created = createDropdown(el);
          break;
        case Type::MULTI_SLIDER:
          created = createMultiSlider(el);
          break;
      }
      QWidget* widget = created.widget;
      QLabel* name = el.type == Type::CHECKBOX || el.type == Type::BUTTON || el.name.empty() ? nullptr : new QLabel(QString::fromStdString(el.name)); // Checkboxes and buttons show their own text, an empty name means no caption
      widget->setToolTip(QString::fromStdString(el.description));
      if (name) {
        name->setToolTip(widget->toolTip());
      }
      if (!section || el.horizontal != sectionHorizontal) {
        sectionHorizontal = el.horizontal;
        if (sectionHorizontal) {
          section = new FlowLayout(nullptr, 3, false); // More space between the elements than between the rows
        } else {
          auto* form = new QFormLayout;
          form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter); // Set explicitly, the defaults depend on the style: e.g. Breeze centers the form and aligns labels right
          form->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
          form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
          form->setRowWrapPolicy(QFormLayout::DontWrapRows);
          section = form;
        }
        section->setContentsMargins(0, 0, 0, 0);
        boxLayout->addLayout(section);
      }
      if (!sectionHorizontal) {
        name ? ((QFormLayout*)section)->addRow(name, widget) : ((QFormLayout*)section)->addRow(widget);
      } else if (name) { // Label and element as one item of the flow layout
        auto* item = new QWidget;
        auto* row = new QHBoxLayout(item);
        row->setContentsMargins(0, 0, 0, 0);
        row->addWidget(name);
        row->addWidget(widget);
        section->addWidget(item);
      } else {
        section->addWidget(widget);
      }
      mElements.push_back({std::move(created.set), std::move(created.setOptions), el.get, el.dynamicOptions, std::nullopt}); // Element ids are sequential
    }
    framesLayout->addWidget(box);
    if (std::any_of(frame.elements.begin(), frame.elements.end(), [](const auto& el) { return (el.type == Type::SLIDER && !fixedLengthSlider(el)) || el.type == Type::MULTI_SLIDER; })) { // Sliders range to the frame border, frame limited to the window width by the layout
      mScaleSetters.emplace_back([this, framesLayout, box](float scale) { framesLayout->setPreferredWidth(box, std::lround(screen()->availableGeometry().width() / 3 * scale)); });
    }
  }
  mScaleSetters.emplace_back([framesLayout](float) { framesLayout->invalidate(); });
  return frames;
}

void GPUDisplayGUI::setScale(float scale)
{
  mScale = clampScale(scale);
  for (const auto& setter : mScaleSetters) {
    setter(mScale);
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
  if (watched == mStatusLabel && event->type() == QEvent::Resize) {
    fitStatusLabel();
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
#ifdef Q_OS_WIN
  if (event->nativeScanCode() == 0x29) { // Key left of 1: console key (^), independent of the layout
#else
  if (event->nativeScanCode() == 49) { // Key left of 1 (XKB keycode): console key (^), independent of the layout
#endif
    return '^';
  }
  switch (event->key()) {
    case Qt::Key_Up:
      return GPUDisplayFrontendKeyVars::KEY_UP;
    case Qt::Key_Down:
      return GPUDisplayFrontendKeyVars::KEY_DOWN;
    case Qt::Key_Left:
      return GPUDisplayFrontendKeyVars::KEY_LEFT;
    case Qt::Key_Right:
      return GPUDisplayFrontendKeyVars::KEY_RIGHT;
    case Qt::Key_PageUp:
      return GPUDisplayFrontendKeyVars::KEY_PAGEUP;
    case Qt::Key_PageDown:
      return GPUDisplayFrontendKeyVars::KEY_PAGEDOWN;
    case Qt::Key_Home:
      return GPUDisplayFrontendKeyVars::KEY_HOME;
    case Qt::Key_End:
      return GPUDisplayFrontendKeyVars::KEY_END;
    case Qt::Key_Insert:
      return GPUDisplayFrontendKeyVars::KEY_INSERT;
    case Qt::Key_Escape:
      return GPUDisplayFrontendKeyVars::KEY_ESCAPE;
    case Qt::Key_Backspace:
      return GPUDisplayFrontendKeyVars::KEY_BACKSPACE;
    case Qt::Key_Delete:
      return GPUDisplayFrontendKeyVars::KEY_DELETE;
    case Qt::Key_Return:
    case Qt::Key_Enter:
      return GPUDisplayFrontendKeyVars::KEY_ENTER;
    default:
      break;
  }
  if (event->key() >= Qt::Key_F1 && event->key() <= Qt::Key_F12) {
    static_assert(GPUDisplayFrontendKeyVars::KEY_F12 - GPUDisplayFrontendKeyVars::KEY_F1 == Qt::Key_F12 - Qt::Key_F1);
    return GPUDisplayFrontendKeyVars::KEY_F1 + (event->key() - Qt::Key_F1);
  }
  const QString text = event->text();
  if (text.size() == 1 && text[0].unicode() >= 32 && text[0].unicode() < 127) { // Printable ASCII, with shift applied
    return text[0].unicode();
  }
  return -1;
}
