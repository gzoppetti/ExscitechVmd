#include "Exscitech/Display/ToggleButton.hpp"

#include <string>
#include <cstdio>

namespace Exscitech
{

  ToggleButton::ToggleButton(int id, const std::string& title,
     QWidget* parent) :
      QPushButton (QString(title.c_str()), parent), m_id (id)
  {
    this->setSelected(false);
  }


  void ToggleButton::setSelected(bool isSelected)
  {
    QPalette pallette = this->palette();
    if(isSelected)
    {
      pallette.setColor(QPalette::Button, Qt::darkYellow);
    }
    else
    {
      pallette.setColor(QPalette::Button, Qt::darkGray);
    }
    this->setPalette(pallette);
    this->update();
  }

  int
  ToggleButton::getId()
  {
    return (m_id);
  }

}

