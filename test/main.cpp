#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "ppm.h"
#include "cudaRenderer.h"
using namespace std;

// collect.cu
void collect_create(int a, int b);

// jet.cu
void jet_setup(int cnt);

int main(int argc, char** argv)
{
	int imageWidth = 1024;
	int imageHeight = 512;
	int particleCount = 50000;

	CudaRenderer* renderer;
	renderer = new CudaRenderer();

  collect_create(particleCount, imageWidth*imageHeight);

	cout << "Hello World!!!" << endl;

	jet_setup(particleCount);


	renderer->allocOutputImage(imageWidth, imageHeight);
    renderer->setupISF();
    renderer->clearImage();
    int colored_pixel = renderer->ISF_locate();
    renderer->ISF_render(colored_pixel);
    writePPMImage(renderer->getImage(), "test.ppm");

	return 0;
}
