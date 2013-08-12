#ifndef MU_GAME_SCROLL_AREA_HPP_
#define MU_GAME_SCROLL_AREA_HPP_

#include <QtCore/QMap>
#include <QtGui/QWidget>
#include <QtGui/QScrollArea>

#include "Exscitech/Display/GameChoiceWidget.hpp"

namespace Exscitech
{
  class GameWidgetScrollArea : public QScrollArea
  {

  public:

    GameWidgetScrollArea (const QMap<QString, GameChoiceWidget*>& gameWidgets, QWidget* parent = NULL);

  protected:

    void
    resizeEvent ( QResizeEvent * event );

  private:

    int m_totalNumberGameWidgets;
    int m_widgetsPerRow;

  private:

    static int ms_stretchFactor;

  };

}

#endif
