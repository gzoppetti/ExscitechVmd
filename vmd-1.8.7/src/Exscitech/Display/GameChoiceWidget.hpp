#ifndef MU_GAME_CHOICE_WIDGET_HPP_
#define MU_GAME_CHOICE_WIDGET_HPP_

#include <string>

#include <QtGui/QWidget>
#include <QtGui/QLabel>

#include "Exscitech/Games/GameController.hpp"

namespace Exscitech
{
  class GameChoiceWidget : public QWidget
  {
    Q_OBJECT

  public:

    GameChoiceWidget (GameController::ExscitechGame gameId, QWidget* parent = NULL);

    void
    drawSelected();

    void
    drawUnselected();

    GameController::ExscitechGame
    getGameId() const;

    void
    doDoubleClick() const;

  protected:

    virtual void
    mouseReleaseEvent ( QMouseEvent * event );

    virtual void
    mouseDoubleClickEvent ( QMouseEvent * event );


  signals:

    void
    gameChoiceSelected();

  public:

    static int
    getWidgetWidth();

  private:

   GameController::ExscitechGame m_game;

   QLabel* m_gameIconLabel;

   static int ms_iconSquareSize;
   static float ms_titleWidthFactor;
   static float ms_highlightBorderWidthFactor;

   static std::string ms_highlightStylesheetText;
   static const std::string ms_defaultIconPath;
  };

}

#endif
