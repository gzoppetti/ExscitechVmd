#include <QtGui/QHBoxLayout>
#include <QtGui/QScrollBar>

#include "Exscitech/Display/ImageSelectionWidget.hpp"

namespace Exscitech
{
  const int imageSize = 120;

  QImage
  scale (const QString &imageFileName)
  {
    QImage image (imageFileName);
    return image.scaled (QSize (imageSize, imageSize), Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation);
  }

  ImageSelectionWidget::ImageSelectionWidget (int iconHeight, int iconWidth, QWidget *parent) :
      QWidget (parent)
  {
    QHBoxLayout* layout = new QHBoxLayout (this);
    m_imageListView = new QListView (this);

    layout->addWidget (m_imageListView);

    m_imageListView->setViewMode (QListView::IconMode);
    m_imageListView->setFlow (QListView::LeftToRight);
    m_imageListView->setWrapping (false);
    m_imageListView->setUniformItemSizes (true);
    m_imageListView->setIconSize (QSize (iconHeight, iconHeight));
    m_imageListView->setSelectionRectVisible (true);
    m_imageListView->setMovement (QListView::Static);
    m_imageListView->setSelectionMode (QListView::SingleSelection);
    m_imageListView->setEditTriggers (QAbstractItemView::NoEditTriggers);

    m_standardModel = new QStandardItemModel (this);
    m_imageListView->setModel (m_standardModel);

    connect (m_imageListView, SIGNAL(clicked(QModelIndex)),
        SLOT(imageClicked(QModelIndex)));
  }

  ImageSelectionWidget::~ImageSelectionWidget ()
  {
    delete m_imageListView;
  }

  void
  ImageSelectionWidget::addToList(int position, const std::string& imageFile, const std::string& text)
  {
    QStandardItem* item = new QStandardItem ();
        item->setIcon (
            QIcon (QPixmap::fromImage (scale (QString(imageFile.c_str())))));
        item->setText (QString(text.c_str()));
    m_standardModel->insertRow(position, item);
  }

  void
  ImageSelectionWidget::push_back(const std::string& imageFile, const std::string& text)
  {
    addToList(m_standardModel->rowCount(), imageFile, text);
  }

  void
  ImageSelectionWidget::push_front(const std::string& imageFile, const std::string& text)
  {
    addToList(0, imageFile, text);
  }

  void
  ImageSelectionWidget::removeFromList(int position)
  {
    m_standardModel->removeRow(position);
  }

  void
  ImageSelectionWidget::pop_front()
  {
    removeFromList(0);
  }

  void
  ImageSelectionWidget::pop_back()
  {
    removeFromList(m_standardModel->rowCount());
  }

  void
  ImageSelectionWidget::clear()
  {
    m_standardModel->clear();
  }

  void
  ImageSelectionWidget::selectIndex(int index)
  {
    //m_imageListView->setCurrentIndex(m_standardModel->index(index, 0));
    m_imageListView->selectionModel()->select(m_standardModel->index(index, 0), QItemSelectionModel::Select);
  }

  void
  ImageSelectionWidget::clearSelection()
  {
    m_imageListView->clearSelection();
  }

  void
  ImageSelectionWidget::imageClicked (QModelIndex index)
  {
      emit selectionChanged (index.row ());
  }
}
