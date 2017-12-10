#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "ppm.h"
#include "cudaRenderer.h"
using namespace std;

void fft();
void collect_test();

void jet_setup(int cnt);

int main(int argc, char** argv)
{
	int imageWidth = 512;
	int imageHeight = 512;
	int particleCount = 5000000;

	CudaRenderer* renderer;
	renderer = new CudaRenderer();

	cout << "Hello World!!!" << endl;

	jet_setup(particleCount);

	// for (int i=0;i<particleCount;i++)
	// {
	// 	printf("%f %f %f\n", particleX[i], particleY[i], particleZ[i]);
	// }

	renderer->allocOutputImage(imageWidth, imageHeight);
    // renderer->loadISF(particleCount, particleX, particleY, particleZ, 2, 2, 2);
    renderer->setupISF();
    renderer->clearImage();
    renderer->ISF_render();
    writePPMImage(renderer->getImage(), "test.ppm");

	return 0;
}
