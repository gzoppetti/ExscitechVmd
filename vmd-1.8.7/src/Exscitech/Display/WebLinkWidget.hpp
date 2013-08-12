#ifndef MU_WEB_LINK_WIDGET_HPP_
#define MU_WEB_LINK_WIDGET_HPP_

#include <string>

#include <QtGui/QWidget>
#include <QtGui/QLabel>

namespace Exscitech
{
  class WebLinkWidget : public QLabel
  {

  public:

    WebLinkWidget (const std::string& title = "", const std::string& linkAddr = "", QWidget* parent = NULL);

    void
    setLinkAddress (const std::string& linkAddr);

    void
    setText(const std::string& text);

  protected:

    virtual void
    mouseReleaseEvent ( QMouseEvent * event );

    virtual void
    enterEvent ( QEvent * event );

    virtual void
    leaveEvent ( QEvent * event );

  private:

   std::string m_linkAddress;
   std::string m_labelText;

  };

}

#endif
