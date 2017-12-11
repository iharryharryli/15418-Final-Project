#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

#define PARTICULE_NUM 5000000
#define AVG_COUNT (PARTICULE_NUM / 250000.0f)


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    int* cudaDevicePixelQueue;
    int cudaDevicePixelQueueSize;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void setupISF();

    void loadISF(int particleCount, double* px, double *py, double *pz,
        int sizex, int sizey, int sizez);

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    //ISF relevant
    void ISF_render(int limit);
    int ISF_locate();

    void render();

    void weak();

    void dumb();

    void smart();

    void best();

    void amazing();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
};


#endif
