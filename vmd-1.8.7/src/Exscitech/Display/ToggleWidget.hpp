#ifndef TOGGLE_WIDGET_HPP_
#define TOGGLE_WIDGET_HPP_

#include <string>
#include <vector>

#include <QtGui/QWidget>

#include "Exscitech/Display/ToggleButton.hpp"

namespace Exscitech
{
  class ToggleWidget : public QWidget
  {
    Q_OBJECT

  public:

    ToggleWidget (const std::vector<std::string>& items, int initialSelected, QWidget* parent = NULL);

    void setSelectedIndex (int id);

    int getSelectedIndex();

  signals:

    void
    itemSelected (int id);

  public slots:

    void
    handleSelectionChange();

  private:

    ToggleButton* m_selected;

  };

}

#endif
