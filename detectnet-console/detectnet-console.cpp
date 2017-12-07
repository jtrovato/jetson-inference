/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "detectNet.h"
#include "loadImage.h"

#include "cudaMappedMemory.h"


#include <sys/time.h>
#include <string>
#include <iostream>
#include "dirent.h"


// main entry point
int main( int argc, char** argv )
{
	printf("detectnet-folder\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc != 2 )
	{
		printf("detectnet-folder:  wrong arguments. use with image folder\n");
		return 0;
	}
	
        const char* Foldername = argv[1];
       
	// create detectNet
        int argc_fake = 3;
        char * argv_fake[4];
        argv_fake[1] = "--prototxt=/home/sarcos/jetson-inference/models/20171201_drone/deploy.prototxt";
        argv_fake[2] = "--model=/home/sarcos/jetson-inference/models/20171201_drone/snapshot_iter_7440.caffemodel";
        argv_fake[3] = NULL;
	detectNet* net = detectNet::Create(argc_fake, argv_fake);

	if( !net )
	{
		printf("detectnet-folder:   failed to initialize detectNet\n");
		return 0;
	}

	net->EnableProfiler();
	
	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
	
        
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (Foldername)) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                printf ("%s\n", ent->d_name);
                const char *imgFilename = ent->d_name; 
       
	
	        if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	        {
	        	printf("failed to load image '%s'\n", imgFilename);
	        	return 0;
	        }
	        
	        // classify image
	        int numBoundingBoxes = maxBoxes;
	        

	        const bool result = net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU);


	        if( !result )
	        	printf("detectnet-console:  failed to classify '%s'\n", imgFilename);
	        else if( argc > 2 )		// if the user supplied an output filename
	        {
	        	printf("%i bounding boxes detected\n", numBoundingBoxes);
	        	
	        	int lastClass = 0;
	        	int lastStart = 0;
	        	
	        	for( int n=0; n < numBoundingBoxes; n++ )
	        	{
	        		const int nc = confCPU[n*2+1];
	        		float* bb = bbCPU + (n * 4);
	        		
	        		printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
	        		
	        		if( nc != lastClass || n == (numBoundingBoxes - 1) )
	        		{
	        			if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
	        				printf("detectnet-console:  failed to draw boxes\n");
	        				
	        			lastClass = nc;
	        			lastStart = n;
	        		}
	        	}
	        	
	        	CUDA(cudaThreadSynchronize());
	        	
	        	// save image to disk
	        	printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
	        	
	        	if( !saveImageRGBA(imgFilename, (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
	        		printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, imgFilename);
	        	else	
	        		printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, imgFilename);
	        	
	        }
	        //printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");


     }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }




	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
