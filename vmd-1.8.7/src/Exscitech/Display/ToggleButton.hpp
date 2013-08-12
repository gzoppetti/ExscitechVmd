#ifndef TOGGLE_BUTTON_HPP_
#define TOGGLE_BUTTON_HPP_

#include <string>

#include <QtGui/QWidget>
#include <QtGui/QPushButton>

namespace Exscitech
{
  class ToggleButton : public QPushButton
  {

  public:

    ToggleButton (int id, const std::string& title = "", QWidget* parent = NULL);

    void
    setSelected (bool isSelected);

    int
    getId();

  private:

   int m_id;

  };

}

#endif
