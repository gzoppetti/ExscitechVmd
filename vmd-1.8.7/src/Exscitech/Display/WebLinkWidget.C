#include "Exscitech/Display/WebLinkWidget.hpp"

#include <string>
#include <cstdio>

#include <QtGui/QLabel>
#include <QtGui/QDesktopServices>
#include <QtCore/QUrl>


namespace Exscitech
{
  WebLinkWidget::WebLinkWidget (const std::string& title, const std::string& linkAddr, QWidget* parent): QLabel(parent), m_linkAddress(linkAddr), m_labelText(title)
  {
    QLabel::setText("<font color='blue'>" + QString(m_labelText.c_str()) + "</font>");
  }

  void
  WebLinkWidget::setLinkAddress(const std::string& linkAddr)
  {
    m_linkAddress = linkAddr;
  }

  void
  WebLinkWidget::setText(const std::string& text)
  {
    m_labelText = text;
    QLabel::setText("<font color='blue'>" + QString(m_labelText.c_str()) + "</font>");
  }

  void
  WebLinkWidget::mouseReleaseEvent ( QMouseEvent * event )
  {
    fprintf(stderr, "following %s\n", m_linkAddress.c_str());
    QUrl linkUrl(QString(m_linkAddress.c_str()));
    QDesktopServices::openUrl(linkUrl);
  }

  void
  WebLinkWidget::enterEvent ( QEvent * event )
  {
    QLabel::setText("<font color='blue'><u>" + QString(m_labelText.c_str()) + "</u></font>");
    setCursor(Qt::PointingHandCursor);
  }

  void
  WebLinkWidget::leaveEvent ( QEvent * event )
  {
    QLabel::setText("<font color='blue'>" + QString(m_labelText.c_str()) + "</font>");
    setCursor(Qt::ArrowCursor);
  }

}

