#ifndef ImageSelectionWidget_HPP_
#define ImageSelectionWidget_HPP_

#include <QtGui/QWidget>
#include <QtGui/QListView>
#include <QtGui/QStandardItemModel>
#include <QtGui/QPushButton>

namespace Exscitech
{
  class ImageSelectionWidget : public QWidget
  {
  Q_OBJECT

  public:
    explicit
    ImageSelectionWidget (int iconHeight, int iconWidth, QWidget *parent = NULL);

    ~ImageSelectionWidget ();

    void
    addToList(int position, const std::string& imageFile, const std::string& text);

    void
    push_back(const std::string& imageFile, const std::string& text);

    void
    push_front(const std::string& imageFile, const std::string& text);

    void
    removeFromList(int position);

    void
    pop_front();

    void
    pop_back();

    void
    clear();

    void
    selectIndex(int index);

    void
    clearSelection();


  signals:
    void
    selectionChanged (int);

  public Q_SLOTS:

    void
    imageClicked (QModelIndex);

  private:

    QListView* m_imageListView;
    QStandardItemModel* m_standardModel;
  };
}

#endif
