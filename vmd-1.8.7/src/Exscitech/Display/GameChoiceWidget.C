#include "Exscitech/Display/GameChoiceWidget.hpp"
#include "Exscitech/Games/GameInfoManager.hpp"

#include <string>

#include <QtGui/QLabel>
#include <QtGui/QBoxLayout>
#include <QtGui/QStyle>
#include <QtGui/QApplication>

namespace Exscitech {
int GameChoiceWidget::ms_iconSquareSize = 100;
float GameChoiceWidget::ms_titleWidthFactor = 1.25f;
float GameChoiceWidget::ms_highlightBorderWidthFactor = 0.1f;

std::string GameChoiceWidget::ms_highlightStylesheetText = "color:red";

const std::string GameChoiceWidget::ms_defaultIconPath =
		"vmd-1.8.7/ExscitechResources/GameResources/DefaultIcon.jpg";

GameChoiceWidget::GameChoiceWidget(GameController::ExscitechGame gameId,
		QWidget* parent) :
		QWidget(parent), m_game(gameId) {
	QVBoxLayout* widgetLayout = new QVBoxLayout();

	m_gameIconLabel = new QLabel();
	m_gameIconLabel->setAlignment(Qt::AlignCenter);
	std::string iconPath = GameInfoManager::getGameIconPath(m_game);
	QPixmap gameIcon(iconPath.c_str());
	if (gameIcon.isNull()) {
		gameIcon.load(QString(ms_defaultIconPath.c_str()));
	}
	m_gameIconLabel->setPixmap(
			gameIcon.scaled(ms_iconSquareSize, ms_iconSquareSize,
					Qt::KeepAspectRatio));
	int highlightMargin = ms_highlightBorderWidthFactor * ms_iconSquareSize;
	m_gameIconLabel->setFixedSize(ms_iconSquareSize + highlightMargin,
			ms_iconSquareSize + highlightMargin);
	widgetLayout->addWidget(m_gameIconLabel, 0, Qt::AlignCenter);

	std::string gameTitle = GameInfoManager::getGameTitle(m_game);
	QLabel* gameTitleLabel = new QLabel(gameTitle.c_str());
	gameTitleLabel->setFixedWidth(ms_iconSquareSize * ms_titleWidthFactor);
	gameTitleLabel->setAlignment(Qt::AlignCenter);
	gameTitleLabel->setWordWrap(true);
	widgetLayout->addWidget(gameTitleLabel, 0, Qt::AlignCenter);
	this->setLayout(widgetLayout);
}

GameController::ExscitechGame GameChoiceWidget::getGameId() const {
	return (m_game);
}

void GameChoiceWidget::drawSelected() {
	m_gameIconLabel->setLineWidth(
			ms_iconSquareSize * ms_highlightBorderWidthFactor * 0.5f);
	m_gameIconLabel->setFrameShape(QFrame::Box);
	m_gameIconLabel->setStyleSheet(QString(ms_highlightStylesheetText.c_str()));
}

void GameChoiceWidget::drawUnselected() {
	m_gameIconLabel->setFrameShape(QFrame::NoFrame);
}

void GameChoiceWidget::doDoubleClick() const {
	static GameController* instance = GameController::acquire();
	instance->startNewGame(m_game);
}

int GameChoiceWidget::getWidgetWidth() {
	int width = ms_iconSquareSize * ms_titleWidthFactor;
	width += 2
			* QApplication::style()->pixelMetric(
					QStyle::PM_LayoutHorizontalSpacing);
	width += QApplication::style()->pixelMetric(QStyle::PM_LayoutRightMargin);
	return (width);
}

void GameChoiceWidget::mouseReleaseEvent(QMouseEvent * event) {
	emit gameChoiceSelected();
}

void GameChoiceWidget::mouseDoubleClickEvent(QMouseEvent * event) {
	doDoubleClick();
}

}

