#include "Exscitech/Display/ToggleWidget.hpp"

#include <string>

#include <QtGui/QBoxLayout>

namespace Exscitech
{

  ToggleWidget::ToggleWidget (const std::vector<std::string>& items,
      int initialSelected, QWidget* parent) :
      QWidget (parent)
  {
    QHBoxLayout* widgetLayout = new QHBoxLayout ();
    widgetLayout->setContentsMargins (0, 0, 0, 0);
    widgetLayout->setSpacing (0);

    QFontMetrics buttonMetrics = QPushButton ().fontMetrics ();

    // calculate width two ways, pick smaller
    // first allows optimum fit for small, similarly sized words
    // second ensures reasonable behavior for widely varied word sizes
    int maxElementWidthCalculation = 0;
    int totalElementWidthsCalculation = 0;
    for (int i = 0; i < items.size (); ++i)
    {
      std::string title = items[i];
      ToggleButton* itemButton = new ToggleButton (i, title);
      itemButton->setFocusPolicy (Qt::NoFocus);

      maxElementWidthCalculation = std::max (
          buttonMetrics.width (QString::fromStdString (title)),
          maxElementWidthCalculation);
      totalElementWidthsCalculation += itemButton->sizeHint ().width ();

      QObject::connect (itemButton, SIGNAL (clicked ()), this,
          SLOT (handleSelectionChange()));
      widgetLayout->addWidget (itemButton, 1);
    }

    maxElementWidthCalculation = (maxElementWidthCalculation + 6)
        * items.size ();
    int widgetWidth = std::min (maxElementWidthCalculation,
        totalElementWidthsCalculation);
    this->setFixedWidth (widgetWidth);

    this->setLayout (widgetLayout);

    m_selected =
        (ToggleButton*) widgetLayout->itemAt (initialSelected)->widget ();
    m_selected->setSelected (true);
  }

  void
  ToggleWidget::setSelectedIndex (int id)
  {
    int previouslySelected = m_selected->getId ();
    if (previouslySelected != id)
    {
      m_selected->setSelected (false);

      m_selected = (ToggleButton*) this->layout ()->itemAt (id)->widget ();
      m_selected->setSelected (true);
      emit itemSelected (id);
    }
  }

  int
  ToggleWidget::getSelectedIndex ()
  {
    return (m_selected->getId ());
  }

  void
  ToggleWidget::handleSelectionChange ()
  {
    ToggleButton* selected = (ToggleButton*) QObject::sender ();
    int newId = selected->getId ();

    int previouslySelected = m_selected->getId ();
    if (previouslySelected != newId)
    {
      m_selected->setSelected (false);

      m_selected = selected;
      m_selected->setSelected (true);
      emit itemSelected (newId);
    }
  }

}

