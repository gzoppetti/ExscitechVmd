#include <sstream>

#include <GL/glew.h>

#include "VMDApp.h"

#include "Exscitech/Types.hpp"
#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Graphics/Scene.hpp"

namespace Exscitech {
Scene::Scene() {
}

Scene::~Scene() {
	for (NameDrawableConstIter i = m_drawables.begin(); i != m_drawables.end();
			++i) {
		delete i->second;
	}

	for (NameDisplayableConstIter i = m_displayables.begin();
			i != m_displayables.end(); ++i) {
		delete i->second;
	}
}

void Scene::addDrawable(const std::string& name, Drawable* const drawable) {
	m_drawables.insert(std::make_pair(name, drawable));
}

void Scene::addDisplayable(const std::string& name,
		Displayable* const displayable) {
	m_displayables.insert(std::make_pair(name, displayable));
}

Drawable*
Scene::removeDrawable(const std::string& name) {
	NameDrawableIter iter = m_drawables.find(name);
	if (iter != m_drawables.end()) {
		Drawable* returnDrawable = iter->second;
		m_drawables.erase(iter);
		return returnDrawable;
	}
	return NULL;
}

Displayable*
Scene::removeDisplayable(const std::string& name) {
	NameDisplayableIter iter = m_displayables.find(name);
	if (iter != m_displayables.end()) {
		Displayable* returnDisplayable = iter->second;
		m_displayables.erase(iter);
		return returnDisplayable;
	}
	return NULL;
}

void Scene::clear() {
	m_drawables.clear();
	m_displayables.clear();

}

void Scene::render(Camera* camera) const {
	static GameController* instance = GameController::acquire();

	for (NameDrawableConstIter i = m_drawables.begin(); i != m_drawables.end();
			++i) {
		i->second->draw(camera);
	}

	for (NameDisplayableConstIter i = m_displayables.begin();
			i != m_displayables.end(); ++i) {
		i->second->on();
		i->second->draw(instance->m_vmdApp->display);
		i->second->off();
	}
}
}
