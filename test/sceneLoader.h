#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__

#include "circleRenderer.h"

void
loadISFScene(
    int numCircles,
    double* px, double *py, double *pz,
    int sizex, int sizey, int sizez,
    float*& position,
    float*& velocity,
    float*& color,
    float*& radius);

void
loadCircleScene(
    SceneName sceneName,
    int& numCircles,
    float*& position,
    float*& velocity,
    float*& color,
    float*& radius);

#endif
