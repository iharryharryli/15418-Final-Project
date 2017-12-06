#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "jet.h"
#include "cudaRenderer.h"
#include "ppm.h"

using namespace std;

void fft();

int main(int argc, char** argv)
{
	int imageWidth = 1024;
	int imageHeight = 1024;
	int particleCount = 500;
	double particleX[particleCount];
	double particleY[particleCount];
	double particleZ[particleCount];

	CudaRenderer* renderer;
	renderer = new CudaRenderer();

	cout << "Hello World!!!" << endl;
	
	jet_setup(particleX, particleY, particleZ, particleCount);
	
	for (int i=0;i<particleCount;i++)
	{
		printf("%f %f %f\n", particleX[i], particleY[i], particleZ[i]);
	}

	renderer->allocOutputImage(imageWidth, imageHeight);
    renderer->loadISF(particleCount, particleX, particleY, particleZ, 2, 2, 2);
    renderer->setupISF();
    renderer->clearImage();
    renderer->render();
    writePPMImage(renderer->getImage(), "test.ppm");

	return 0;
}
