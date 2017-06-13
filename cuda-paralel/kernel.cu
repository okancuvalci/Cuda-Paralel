#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#define Rp 10001
#define Clm 99911
#define veri 99911
#define Width 10000
#define BLOCKSIZE 125


//Bloklara böldüðümüz matris ve vektörün çarpýmý//
__global__ void carpim(float *Md, float *Nd, float *Pd)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (tid<Width){
		for (int i = 0; i<Width; i++)
			sum += Md[i + Width*tid] * Nd[i];
		Pd[tid] = sum;
	}
}
int main()
{
	clock_t start, end;
	start = clock();
	FILE *satir = fopen("Rp.txt", "r");
	FILE *sutun = fopen("C.txt", "r");
	FILE *data = fopen("D.txt", "r");
	int i = 0, s = 0, l = 0, z = 0;
	static int A[Rp], B[Rp - 1], C[Clm];
	static float D[veri];
	static float M[Width][Width], N[Width][1], sonuc[Width][1];
	//N matrisi oluþturuluyor//
	for (int i = 0; i < Width; i++)
	{
		N[i][0] = 1;
	}
	//Rp.txt dosyasýndan okunan deðerler A matrisine atýlýyor//
	for (i = 0; i < Rp; i++)
	{
		fscanf(satir, "%d", &A[i]);
	}
	//A matrisindeki deðerlerin sonucu B matrisine aktarýlýyor//
	for (i = 0; i < Rp - 1; i++)
	{
		B[i] = A[i + 1] - A[i];
	}
	//C.txt dosyasýndaki sutun bilgileri C matrisine atýlýyor//
	for (i = 0; i < Clm; i++)
	{
		fscanf(sutun, "%d", &C[i]);
	}
	//D.txt dosyasýndaki veriler D matrisine aktarýlýyor//
	for (i = 0; i < veri; i++)
	{
		fscanf(data, "%f", &D[i]);

	}
	//M matrisinin tüm elemanlarýný 0'a setliyoruz//
	for (int i = 0; i < Width; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			M[i][j] = 0;
		}
	}
	//Dosyalardan okuyup sýrayla B C D matrislerine verdiðim deðerlere göre//
	//iþlem yapacagým 10000*10000'lik M matrisi oluþturuluyor//
	for (i = 0; i < Rp - 1; i++)
	{
		for (l = 0; l < B[i]; l++)
		{
			z = C[s];
			M[i][z] = D[s];
			s++;
		}
	}
	float *Md, *Nd, *Pd;
	float *d_Md, *d_Nd, *d_Pd;
	int a = 0, b = 0;
	//CPU da Md Nd ve Pd deðiþkenleri için alan tahsisi//
	Md = (float*)malloc(Width * Width * sizeof(float));
	Nd = (float*)malloc(Width * 1 * sizeof(float));
	Pd = (float*)malloc(Width * 1 * sizeof(float));
	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			Md[a] = M[i][j];
			a++;
		}
	}
	for (int i = 0; i < Width; i++) {
		Nd[i] = 1;
	}
	//GPU da d_Md, d_Nd, d_Pd deðiþkenleri için alan tahsisi//
	cudaMalloc((void**)&d_Md, Width * Width * sizeof(float));
	cudaMalloc((void**)&d_Nd, Width * 1 * sizeof(float));
	cudaMalloc((void**)&d_Pd, Width * 1 * sizeof(float));
	//CPU daki verilerin GPU belleðine atýlmasý//
	cudaMemcpy(d_Md, Md, Width * Width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Nd, Nd, Width * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pd, Pd, Width * 1 * sizeof(float), cudaMemcpyHostToDevice);
	//matrisi böleceðimiz bloklarýn ve koþacak threadlarýn setlenmesi//
	dim3 dimGrid(Width / BLOCKSIZE, Width / BLOCKSIZE);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	//parametlerin fonksiyona verilmesi//
	carpim << <Width / BLOCKSIZE, BLOCKSIZE >> >(d_Md, d_Nd, d_Pd);
	//geri dönen d_Pd sonuc deðerinin GPU belleðinden CPU belleðine aktarýlmasý//
	cudaMemcpy(Pd, d_Pd, Width * 1 * sizeof(float), cudaMemcpyDeviceToHost);
	//Paralel olarak hesaplanan ilk 10 degerin ekrana basýlmasý//
	printf("ASAGIDA PROGRAM CIKTISININ ILK 10 DEGERI VERILMISTIR\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%d. deger --> %.3f\n", i + 1, Pd[i]);
	}
	end = clock();
	//paralel süre hesaplanmasý ve ekrana basýlmasý//
	printf("Paralel Hesaplama Suresi --> %f(sn)\n", (((float)end - (float)start) / 1000000.0F) * 1000);
	free(Md);
	free(Nd);
	free(Pd);
	cudaFree(d_Md);
	cudaFree(d_Nd);
	cudaFree(d_Pd);
	return 0;
}