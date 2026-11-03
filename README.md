# SkylineOperatorHPC
Progetto d'esame sulla paralelizzazione dello Skyline operator con OpenMP e Cuda.  
Reupload del 2025.   

## Specifica del progetto  
Scopo del progetto è la realizzazione di due implementazioni parallele dell'algoritmo per il calcolo
dello skyline. Vi sono due implementazioni: uno in OpenMP e uno in Cuda.

## Uso  
Il programma viene lanciato dalla riga di comando nel modo seguente:

### Linux:
./omp-skyline.c < input_file > output_file

### Windows powershell:  
& cmd.exe /c ".\omp-skyline.c < input_file > out"  

### Cuda:  
nvcc .\cuda-skyline.cu -o .\input_file    

Il programma scrive il risultato su standard output.

##
Nella directory **datasets/** vengono forniti alcuni file di input che è possibile utilizzare per testare
l’algoritmo. Le caratteristiche dei file di input sono riassunte nella seguente tabella:  

| File  | Numero punti (N)  | Dimensioni (D) | Punti skyline (K) |  
|----------|:-------------:|:-------------:|------:|  
| circle | 1000 | 2 | 15 |  
| test1 | 100000 | 3 | 26 |  
| test2 | 100000 | 4 | 10352 |  
| test3* | 100020 | 10 | 24892 |  
| test4* | 100009 | 8 | 17458 |  
| test5* | 100000 | 20 | 96973 |  
| test6* | 100100 | 50 | 100050 |  
| test7* | 100400 | 200 | 100200 |  
| worst | 100000 | 10 | 100000 |  

*Escusi per via delle dimensioni.