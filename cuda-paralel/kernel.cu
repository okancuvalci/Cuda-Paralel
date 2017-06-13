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


//Bloklara b�ld���m�z matris ve vekt�r�n �arp�m�//
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
	//N matrisi olu�turuluyor//
	for (int i = 0; i < Width; i++)
	{
		N[i][0] = 1;
	}
	//Rp.txt dosyas�ndan okunan de�erler A matrisine at�l�yor//
	for (i = 0; i < Rp; i++)
	{
		fscanf(satir, "%d", &A[i]);
	}
	//A matrisindeki de�erlerin sonucu B matrisine aktar�l�yor//
	for (i = 0; i < Rp - 1; i++)
	{
		B[i] = A[i + 1] - A[i];
	}
	//C.txt dosyas�ndaki sutun bilgileri C matrisine at�l�yor//
	for (i = 0; i < Clm; i++)
	{
		fscanf(sutun, "%d", &C[i]);
	}
	//D.txt dosyas�ndaki veriler D matrisine aktar�l�yor//
	for (i = 0; i < veri; i++)
	{
		fscanf(data, "%f", &D[i]);

	}
	//M matrisinin t�m elemanlar�n� 0'a setliyoruz//
	for (int i = 0; i < Width; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			M[i][j] = 0;
		}
	}
	//Dosyalardan okuyup s�rayla B C D matrislerine verdi�im de�erlere g�re//
	//i�lem yapacag�m 10000*10000'lik M matrisi olu�turuluyor//
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
	//CPU da Md Nd ve Pd de�i�kenleri i�in alan tahsisi//
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
	//GPU da d_Md, d_Nd, d_Pd de�i�kenleri i�in alan tahsisi//
	cudaMalloc((void**)&d_Md, Width * Width * sizeof(float));
	cudaMalloc((void**)&d_Nd, Width * 1 * sizeof(float));
	cudaMalloc((void**)&d_Pd, Width * 1 * sizeof(float));
	//CPU daki verilerin GPU belle�ine at�lmas�//
	cudaMemcpy(d_Md, Md, Width * Width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Nd, Nd, Width * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pd, Pd, Width * 1 * sizeof(float), cudaMemcpyHostToDevice);
	//matrisi b�lece�imiz bloklar�n ve ko�acak threadlar�n setlenmesi//
	dim3 dimGrid(Width / BLOCKSIZE, Width / BLOCKSIZE);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	//parametlerin fonksiyona verilmesi//
	carpim << <Width / BLOCKSIZE, BLOCKSIZE >> >(d_Md, d_Nd, d_Pd);
	//geri d�nen d_Pd sonuc de�erinin GPU belle�inden CPU belle�ine aktar�lmas�//
	cudaMemcpy(Pd, d_Pd, Width * 1 * sizeof(float), cudaMemcpyDeviceToHost);
	//Paralel olarak hesaplanan ilk 10 degerin ekrana bas�lmas�//
	printf("ASAGIDA PROGRAM CIKTISININ ILK 10 DEGERI VERILMISTIR\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%d. deger --> %.3f\n", i + 1, Pd[i]);
	}
	end = clock();
	//paralel s�re hesaplanmas� ve ekrana bas�lmas�//
	printf("Paralel Hesaplama Suresi --> %f(sn)\n", (((float)end - (float)start) / 1000000.0F) * 1000);
	free(Md);
	free(Nd);
	free(Pd);
	cudaFree(d_Md);
	cudaFree(d_Nd);
	cudaFree(d_Pd);
	return 0;
}