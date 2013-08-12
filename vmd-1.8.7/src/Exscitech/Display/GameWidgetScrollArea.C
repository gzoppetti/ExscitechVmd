#include <cstdio>
#include <algorithm>

#include <QtGui/QKeyEvent>
#include <QtGui/QBoxLayout>

#include "Exscitech/Display/GameWidgetScrollArea.hpp"

namespace Exscitech {
int GameWidgetScrollArea::ms_stretchFactor = 100000;

GameWidgetScrollArea::GameWidgetScrollArea(
		const QMap<QString, GameChoiceWidget*>& gameWidgets, QWidget* parent) :
		QScrollArea(parent), m_totalNumberGameWidgets(gameWidgets.size()), m_widgetsPerRow(
				m_totalNumberGameWidgets) {
	QWidget* content = new QWidget();
	QVBoxLayout* overallLayout = new QVBoxLayout();
	content->setLayout(overallLayout);

	QHBoxLayout* initialOneLine = new QHBoxLayout();

	foreach(GameChoiceWidget * gameWidget, gameWidgets)
	initialOneLine->addWidget(gameWidget, 1, Qt::AlignLeading);

	initialOneLine->addStretch(ms_stretchFactor);

	overallLayout->addLayout(initialOneLine);

	overallLayout->addStretch(ms_stretchFactor);

	this->setWidget(content);

	this->setWidgetResizable(true);
	this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void GameWidgetScrollArea::resizeEvent(QResizeEvent * event) {
	int widgetWidth = GameChoiceWidget::getWidgetWidth();
	int widgetsPerRowNew = event->size().width() / widgetWidth;

	if (widgetsPerRowNew <= 0) {
		fprintf(stderr, "Error: window too small. Exiting.\n");
		static GameController* instance = GameController::acquire();
		instance->terminateApplication();
	}

	if (widgetsPerRowNew < m_widgetsPerRow) {
		// too many widgets per row, must make rows shorter

		QVBoxLayout* overallLayout = (QVBoxLayout*) this->widget()->layout();

		int numToMove = m_widgetsPerRow - widgetsPerRowNew;
		for (int i = 0; i < overallLayout->count() - 1 && numToMove > 0; ++i) {
			QHBoxLayout* horizontalRow =
					(QHBoxLayout*) overallLayout->itemAt(i)->layout();
			QHBoxLayout* nextHorizontalRow;
			for (int j = 0; j < numToMove; ++j) {
				QWidget* lastWidget = horizontalRow->itemAt(
						horizontalRow->count() - 2)->widget();
				horizontalRow->removeWidget(lastWidget);

				if (i == overallLayout->count() - 2) {
					QHBoxLayout* newRow = new QHBoxLayout();
					newRow->addStretch(ms_stretchFactor);
					overallLayout->insertLayout(overallLayout->count() - 1,
							newRow);
				}

				nextHorizontalRow =
						(QHBoxLayout*) overallLayout->itemAt(i + 1)->layout();
				nextHorizontalRow->insertWidget(0, lastWidget, 0,
						Qt::AlignLeading);
			}

			numToMove = nextHorizontalRow->count() - widgetsPerRowNew - 1;
		}

		m_widgetsPerRow = std::min(widgetsPerRowNew, m_totalNumberGameWidgets);
	} else if (m_widgetsPerRow < m_totalNumberGameWidgets
			&& widgetsPerRowNew > m_widgetsPerRow) {
		// add widgets to each row to make each longer and have less rows

		QVBoxLayout* overallLayout = (QVBoxLayout*) this->widget()->layout();

		int numToMove = widgetsPerRowNew - m_widgetsPerRow;

		QHBoxLayout* currentHorizontalRow =
				(QHBoxLayout*) overallLayout->itemAt(0)->layout();
		for (int i = 1; i < overallLayout->count() - 1; ++i) {
			QHBoxLayout* nextHorizontalRow =
					(QHBoxLayout*) overallLayout->itemAt(i)->layout();

			for (int j = 0; j < numToMove; ++j) {
				if (nextHorizontalRow->count() <= 1) {
					if (i == overallLayout->count() - 2) {
						overallLayout->removeItem(overallLayout->itemAt(i));
						break;
					}
					overallLayout->removeItem(overallLayout->itemAt(i));
					nextHorizontalRow =
							(QHBoxLayout*) overallLayout->itemAt(i)->layout();
				}
				QWidget* firstWidget = nextHorizontalRow->itemAt(0)->widget();
				nextHorizontalRow->removeWidget(firstWidget);
				currentHorizontalRow->insertWidget(
						currentHorizontalRow->count() - 1, firstWidget);
			}

			currentHorizontalRow = nextHorizontalRow;

			numToMove = widgetsPerRowNew - (nextHorizontalRow->count() - 1);
		}

		m_widgetsPerRow = std::min(widgetsPerRowNew, m_totalNumberGameWidgets);
	}

	QWidget* content = this->widget();
	content->resize(this->width(), content->sizeHint().height());
}
}

