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
void jet_main(int i);

// particle.cu
void tpstart(int a);
void tpend(int a);
void tpsummary();
void tpinit();

int main(int argc, char** argv)
{

  if(argc < 2)
  {
    cout << "no argument!\n" << endl;
    return 0;
  }

  int frame_gen_index = atoi(argv[1]);

	int imageWidth = 1024;
	int imageHeight = 512;
	int particleCount = PARTICULE_NUM;

	CudaRenderer* renderer;
	renderer = new CudaRenderer();


  collect_create(particleCount, imageWidth*imageHeight);
  
  renderer->allocOutputImage(imageWidth, imageHeight);
  renderer->setupISF();

	cout << "Hello World!!!" << endl;

	jet_setup(particleCount);

  tpinit();

  char filename_buf[100];

  for(int i=0; i<1000; i++)
  {
    jet_main(i);

    tpstart(18);
    renderer->clearImage();
    renderer->ISF_Fast_Render();
    tpend(18);
    
    if(i > frame_gen_index){
    sprintf(filename_buf,"frame/%d.ppm", i);
    writePPMImage(renderer->getImage(), filename_buf);
    }

    printf("iteration %d done. \n", i);

  }

  tpsummary();

	return 0;
}
